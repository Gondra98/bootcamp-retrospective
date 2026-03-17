# ================================================================
#  app.py  —  Flask 메인 애플리케이션
#  드론 비행 관리 시스템의 핵심 파일.
#  라우팅(URL → 함수 연결), DB 접속, 날씨 API, 지도 API 등을 담당.
# ================================================================

# 패키지 설치 안내 (최초 1회만 필요)
# pip install python-dotenv  → .env 파일 읽기
# pip install pymysql        → MariaDB/MySQL 연결
# pip install astral         → 일출/일몰 시각 계산

# .env 파일에서 환경변수를 읽어오는 라이브러리
from dotenv import load_dotenv

# Flask 웹 프레임워크에서 필요한 기능들을 불러옴
# Flask            - 앱 객체 생성
# render_template  - HTML 파일 렌더링(화면 출력)
# request          - 클라이언트(브라우저)가 보낸 데이터 읽기
# redirect         - 다른 URL로 이동
# url_for          - 함수 이름으로 URL 생성
# flash            - 1회성 메시지(로그인 성공/실패 등)
# get_flashed_messages - flash 메시지 꺼내기
# session          - 로그인 정보 등 브라우저 세션 관리
# jsonify          - 딕셔너리를 JSON 응답으로 변환
from flask import Flask, render_template, request, redirect, url_for, flash, get_flashed_messages, session, jsonify

# MariaDB/MySQL에 접속하기 위한 드라이버
import pymysql

# 운영체제 환경변수, 파일 경로 처리
import os

# 파일 업로드 시 안전한 파일명으로 변환 (경로 조작 공격 방지)
from werkzeug.utils import secure_filename

# 이미지 처리 라이브러리 (업로드 이미지 검증 등에 활용)
from PIL import Image

# 날짜/시간 처리 → 스케줄러, 기상 데이터 시각 포맷에 사용
from datetime import datetime, timedelta

# 기상청 API 호출을 위한 HTTP 클라이언트
import requests

# 일출/일몰 계산 라이브러리 — 야간 비행 금지 판단에 사용
from astral import LocationInfo
from astral.sun import sun

# 시간대 처리 (한국 = Asia/Seoul)
import pytz

# 날씨 스케줄러 루프에서 잠시 대기할 때 사용
import time

# 날씨 자동 업데이트를 백그라운드에서 돌리기 위한 스레드
import threading

# 서버 종료 시 실행할 정리 함수 등록 (DB 초기화 등)
import atexit

# HTTP 요청 재시도 전략 설정에 필요한 클래스들
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# .env 파일의 내용을 환경변수로 로드 (DB 접속 정보, API KEY 등)
load_dotenv()

# Flask 앱 인스턴스 생성
app = Flask(__name__)

# 세션 암호화에 사용하는 비밀키
# .env에 FLASK_SECRET_KEY가 있으면 그 값을, 없으면 랜덤 값을 사용
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24)

# ── DB 접속 정보 — .env 파일에서 읽어옴 ──────────────────────
DB_HOST     = os.getenv("DB_HOST")                        # DB 서버 주소
DB_PORT     = int(os.getenv("DB_PORT"))                   # DB 포트 (기본 3306)
DB_USER     = os.getenv("DB_USER")                        # DB 계정명
DB_PASSWORD = os.getenv("DB_PASSWORD")                    # DB 비밀번호
DB_NAME     = os.getenv("DB_NAME", "airspace_db")         # 일반 DB 이름
DB_AIRSPACE = os.getenv("DB_AIRSPACE", "airspace_db")     # 공역(지도) 전용 DB 이름
API_KEY_WEA  = os.getenv("API_KEY_WEA")                   # 기상청 지상 관측 API 키
API_KEY_WEAS = os.getenv("API_KEY_WEAS")                  # 기상청 해상 관측 API 키

# ── 파일 업로드 설정 ──────────────────────────────────────────
# 업로드 파일이 저장될 실제 폴더 경로 (static/uploads/)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# 허용되는 파일 확장자 목록 (보안상 제한)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}

# 폴더가 없으면 자동 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """업로드 파일의 확장자가 허용 목록에 있는지 확인하는 함수.
    '.'이 있어야 하고, 마지막 확장자가 ALLOWED_EXTENSIONS 안에 있어야 True."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_conn():
    """일반 DB(사용자, 드론, 허가 신청 등)에 접속해서 연결 객체를 반환.
    charset=utf8mb4   → 한글 + 이모지까지 저장 가능
    DictCursor        → 결과를 row[0] 대신 row['컬럼명']으로 접근
    autocommit=True   → INSERT/UPDATE 후 자동 저장 (commit 생략 가능)"""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


def get_airspace_conn():
    """공역(비행금지구역, 기상관측소 등) 전용 DB에 접속하는 함수.
    get_conn()과 거의 동일하지만 database=DB_AIRSPACE 를 사용."""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_AIRSPACE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


# ── 관리자 Blueprint 등록 ──────────────────────────────────────
# Blueprint: 라우트(URL)를 별도 파일로 분리하는 Flask 기능
# admin_routes.py에서 만든 admin_bp 블루프린트를 메인 앱에 등록
from admin_routes import admin_bp
app.register_blueprint(admin_bp)   # /admin/* 으로 시작하는 URL이 자동 연결됨


# ######################################################################
#                          메인 페이지  /
# ######################################################################
@app.route("/")         # 루트 URL 접속 시 이 함수 실행
def index():
    # Vworld 지도 API 키 가져오기 (지도 화면에서 사용)
    api_key = os.getenv("VWORLD_API_KEY")

    drone_weight = None   # 로그인 안 된 경우 드론 무게 없음

    # 로그인된 유저라면 → DB에서 해당 유저의 드론 최대이륙중량 조회
    if session.get('user_id'):
        try:
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT max_takeoff_weight FROM drone WHERE user_id=%s",
                    (session['user_id'],)
                )
                row = cur.fetchone()
                # 드론 정보가 있으면 float으로 변환
                if row and row['max_takeoff_weight']:
                    drone_weight = float(row['max_takeoff_weight'])
            conn.close()
        except:
            pass   # DB 오류가 나도 메인 페이지는 그냥 열림

    # main.html 템플릿에 api_key와 drone_weight를 전달해서 렌더링
    return render_template("main.html", api_key=api_key, drone_weight=drone_weight)


# ######################################################################
#                       비행가능 확인 페이지  /flight_check
# ######################################################################
@app.route("/flight_check")
def flight_check():
    # 현재는 main.html을 그대로 사용 (추후 별도 템플릿으로 분리 예정)
    return render_template("main.html")


# ######################################################################
#                       비행 허가 페이지  /permit
# ######################################################################
@app.route("/permit")
def permit():
    return render_template("main.html")  # 추후 별도 템플릿으로 분리


# ######################################################################
#                       공지사항 페이지  /notice
# ######################################################################
@app.route("/notice")
def notice():
    return render_template("main.html")  # 추후 별도 템플릿으로 분리


# ######################################################################
#                           회원가입  /register
# ######################################################################
@app.route("/register", methods=["GET", "POST"])
def register():
    # GET 요청: 회원가입 폼 화면 표시
    # POST 요청: 폼 데이터 처리 후 DB 저장

    if request.method == "POST":

        # 폼에서 입력된 기본 회원 정보 가져오기
        name     = request.form["name"]       # 이름
        birth    = request.form["birth"]      # 생년월일
        phone    = request.form["phone"]      # 전화번호
        login_id = request.form["login_id"]   # 로그인 아이디
        password = request.form["password"]   # 비밀번호

        # 드론 정보 — 없으면 None으로 저장 (선택 항목이므로 or None 처리)
        drone_type         = request.form.get("drone_type")         or None
        weight             = request.form.get("weight")             or None
        size               = request.form.get("size")               or None
        max_takeoff_weight = request.form.get("max_takeoff_weight") or None

        conn = get_conn()
        cur  = conn.cursor()

        # 아이디 중복 체크 — 같은 login_id가 이미 있으면 에러 반환
        cur.execute("SELECT user_id FROM users WHERE login_id = %s", (login_id,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return render_template("register.html", error="이미 사용 중인 아이디입니다.")

        # users 테이블에 새 회원 INSERT
        sql_user = """
        INSERT INTO users (login_id,password,name,birth,phone)
        VALUES (%s,%s,%s,%s,%s)
        """
        cur.execute(sql_user, (login_id, password, name, birth, phone))

        # 방금 삽입된 row의 user_id를 가져와서 drone 테이블에도 저장
        user_id = cur.lastrowid

        # 드론 정보 중 하나라도 입력됐다면 drone 테이블에 INSERT
        if any([drone_type, weight, size, max_takeoff_weight]):
            sql_drone = """
            INSERT INTO drone (user_id,drone_type,weight,size,max_takeoff_weight)
            VALUES (%s,%s,%s,%s,%s)
            """
            cur.execute(sql_drone, (user_id, drone_type, weight, size, max_takeoff_weight))

        cur.close()
        conn.close()

        # 1회성 성공 메시지 등록 후 로그인 페이지로 리다이렉트
        flash("회원가입 완료")
        return redirect(url_for("login"))

    # GET 요청이면 빈 회원가입 폼 반환
    return render_template("register.html")


# ######################################################################
#                           로그인  /login
# ######################################################################
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        login_id = request.form["login_id"]
        password = request.form["password"]

        conn = get_conn()
        cur  = conn.cursor()

        # 1단계: 일반 유저 테이블에서 아이디+비밀번호 확인
        cur.execute(
            "SELECT * FROM users WHERE login_id=%s AND password=%s",
            (login_id, password)
        )
        user = cur.fetchone()

        # 2단계: 일반 유저가 아니면 관리자 테이블도 확인
        if not user:
            cur.execute(
                "SELECT * FROM admins WHERE login_id=%s AND password=%s",
                (login_id, password)
            )
            admin = cur.fetchone()
        else:
            admin = None   # 이미 일반 유저로 찾았으면 관리자 불필요

        cur.close()
        conn.close()

        if user:
            # 가입 후 관리자 승인 대기 중인 계정은 로그인 불가
            if user["status"] != "승인":
                flash("관리자 승인 후 로그인 가능합니다.")
            else:
                # 세션에 유저 정보 저장 → 이후 페이지에서 session['user_id'] 등으로 접근 가능
                session["user_id"]  = user["user_id"]
                session["name"]     = user["name"]
                session["is_admin"] = False    # 일반 유저 표시
                flash("로그인 성공")
                return redirect(url_for("index"))

        elif admin:
            # 관리자 세션 저장 후 관리자 대시보드로 이동
            session["admin_id"] = admin["admin_id"]
            session["name"]     = admin["admin_name"]
            session["is_admin"] = True    # 관리자 표시
            flash("관리자로 로그인 되었습니다.")
            return redirect(url_for("admin.admin_dashboard"))

        else:
            # 둘 다 없으면 오류 메시지
            flash("아이디 또는 비밀번호 오류")

    return render_template("login.html")


# 로그아웃  /logout
@app.route("/logout")
def logout():
    session.clear()     # 세션 전체 삭제 (로그인 정보 제거)
    flash("로그아웃 완료")
    return redirect(url_for("index"))


# ######################################################################
#                           마이페이지  /mypage
# ######################################################################
@app.route("/mypage")
def mypage():
    # 로그인 안 된 상태면 로그인 페이지로 보냄
    if "user_id" not in session:
        flash("로그인 필요")
        return redirect(url_for("login"))

    conn = get_conn()
    cur  = conn.cursor()

    # 현재 로그인된 유저 정보 조회
    cur.execute("SELECT * FROM users WHERE user_id = %s", (session["user_id"],))
    user = cur.fetchone()

    # 해당 유저의 드론 정보 조회 (없을 수도 있음)
    cur.execute("SELECT * FROM drone WHERE user_id = %s", (session["user_id"],))
    drone = cur.fetchone()

    cur.close()
    conn.close()

    # 템플릿에 개별 변수로 전달 (drone이 None이면 빈 문자열로 처리)
    return render_template("mypage.html",
        login_id           = user["login_id"],
        name               = user["name"],
        birth              = user["birth"],
        phone              = user["phone"],
        drone_type         = (drone["drone_type"]         if drone else None) or "",
        weight             = (drone["weight"]             if drone else None) or "",
        size               = (drone["size"]               if drone else None) or "",
        max_takeoff_weight = (drone["max_takeoff_weight"] if drone else None) or "",
    )


@app.route("/mypage/update", methods=["POST"])
def mypage_update():
    """마이페이지 정보 수정. 비밀번호는 입력했을 때만 변경."""
    if "user_id" not in session:
        flash("로그인 필요")
        return redirect(url_for("login"))

    # 폼 데이터 수신
    name     = request.form.get("name")
    birth    = request.form.get("birth")
    phone    = request.form.get("phone")
    password = request.form.get("password")   # 빈 문자열이면 변경 안 함

    drone_type         = request.form.get("drone_type")         or None
    weight             = request.form.get("weight")             or None
    size               = request.form.get("size")               or None
    max_takeoff_weight = request.form.get("max_takeoff_weight") or None

    conn = get_conn()
    cur  = conn.cursor()

    # 비밀번호 변경 여부에 따라 쿼리 분기
    if password:
        cur.execute("""
            UPDATE users SET name=%s, birth=%s, phone=%s, password=%s
            WHERE user_id=%s
        """, (name, birth, phone, password, session["user_id"]))
    else:
        cur.execute("""
            UPDATE users SET name=%s, birth=%s, phone=%s
            WHERE user_id=%s
        """, (name, birth, phone, session["user_id"]))

    # 기존 드론 레코드 존재 여부 확인
    cur.execute("SELECT drone_id FROM drone WHERE user_id=%s", (session["user_id"],))
    drone = cur.fetchone()

    # 드론 관련 필드 중 하나라도 값이 있으면 드론 데이터 처리
    has_drone_data = any([drone_type, weight, size, max_takeoff_weight])

    if has_drone_data:
        if drone:
            # 기존 드론 레코드 UPDATE
            cur.execute("""
                UPDATE drone SET drone_type=%s, weight=%s, size=%s, max_takeoff_weight=%s
                WHERE user_id=%s
            """, (drone_type, weight, size, max_takeoff_weight, session["user_id"]))
        else:
            # 드론 레코드가 없으면 새로 INSERT
            cur.execute("""
                INSERT INTO drone (user_id, drone_type, weight, size, max_takeoff_weight)
                VALUES (%s, %s, %s, %s, %s)
            """, (session["user_id"], drone_type, weight, size, max_takeoff_weight))
    else:
        # 드론 정보 전부 비워져 있으면 레코드 삭제
        if drone:
            cur.execute("DELETE FROM drone WHERE user_id=%s", (session["user_id"],))

    cur.close()
    conn.close()

    # 세션의 이름도 최신 값으로 업데이트
    session["name"] = name

    # 저장 완료 알림을 JS alert으로 보여주고 마이페이지로 이동
    return """
        <script>
            alert('✅ 정보가 저장되었습니다.');
            location.href = '/mypage';
        </script>
    """


# ######################################################################
#                  지도 API  /api/zones
#  비행금지·제한·위험구역 폴리곤 좌표를 JSON으로 반환
# ######################################################################

# DB의 zone_category 값 → JS에서 사용하는 타입 문자열 매핑
CATEGORY_MAP = {
    "FORBIDDEN":  "forbidden",    # 완전 비행금지
    "RESTRICTED": "restricted",   # 비행제한 (승인 필요)
    "DANGER":     "danger",       # 위험구역
    "CAUTION":    "danger",       # 주의구역도 위험과 동일 처리
    "SPECIAL":    "restricted",   # 특별구역도 제한으로 분류
}


@app.route("/api/zones")
def api_zones():
    """공역 구역 폴리곤 좌표 목록을 JSON으로 반환.
    OpenLayers 지도에서 구역을 그릴 때 호출됨."""
    try:
        conn = get_airspace_conn()
        with conn.cursor() as cur:
            # 활성화된 구역의 zone 정보 + 좌표 데이터 JOIN 조회
            sql = """
                SELECT
                    z.zone_id, z.zone_name, z.zone_category, z.zone_code,
                    g.polygon_index, g.ring_index, g.point_order,
                    g.longitude, g.latitude
                FROM airspace_zone     z
                JOIN airspace_geometry g ON z.zone_id = g.zone_id
                WHERE z.is_active = 1
                AND z.zone_category IN ('FORBIDDEN', 'RESTRICTED', 'DANGER', 'CAUTION', 'SPECIAL')
                ORDER BY z.zone_id, g.polygon_index, g.ring_index, g.point_order
            """
            cur.execute(sql)
            rows = cur.fetchall()
        conn.close()

        # zone_id + polygon_index 조합을 키로 폴리곤 딕셔너리 생성
        poly_map = {}
        for row in rows:
            key = (row["zone_id"], row["polygon_index"])
            if key not in poly_map:
                poly_map[key] = {
                    "id":       row["zone_id"],
                    "name":     row["zone_name"],
                    "code":     row["zone_code"] or "",
                    "type":     CATEGORY_MAP.get(row["zone_category"], "restricted"),
                    "category": row["zone_category"],
                    "coords":   []    # 좌표 리스트 (순서대로 폴리곤 꼭짓점)
                }
            # 좌표 누적
            poly_map[key]["coords"].append([row["longitude"], row["latitude"]])

        return jsonify({"ok": True, "zones": list(poly_map.values())})

    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/zone_log", methods=["POST"])
def api_zone_log():
    """로그인 유저가 지도에서 클릭한 좌표를 DB에 기록.
    비행 구역 조회 이력 분석에 활용됨."""
    if "user_id" not in session:
        return jsonify({"ok": False, "msg": "로그인 필요"})

    data = request.get_json()   # POST body의 JSON 파싱
    lat  = data.get("latitude")
    lng  = data.get("longitude")

    if lat is None or lng is None:
        return jsonify({"ok": False, "msg": "좌표 누락"})

    try:
        conn = get_conn()
        with conn.cursor() as cur:
            sql = """
                INSERT INTO flight_zone_log (user_id, latitude, longitude)
                VALUES (%s, %s, %s)
            """
            cur.execute(sql, (session["user_id"], lat, lng))
        conn.close()
        return jsonify({"ok": True})

    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/address")
def api_address():
    """클릭한 위경도 → 행정구역명(주소)으로 역지오코딩.
    vworld API 대신 자체 DB 폴리곤으로 구현 (CORS 우회 목적)."""
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "잘못된 좌표"})

    def point_in_polygon(px, py, coords):
        """Ray Casting 알고리즘 — 점(px,py)이 폴리곤 안에 있는지 판별.
        수업에서 배운 알고리즘: 점에서 오른쪽으로 무한히 선을 그어
        폴리곤 변과 교차 횟수가 홀수면 내부, 짝수면 외부."""
        n      = len(coords)
        inside = False
        j      = n - 1
        for i in range(n):
            xi, yi = coords[i]
            xj, yj = coords[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    try:
        from collections import defaultdict
        conn = get_airspace_conn()
        with conn.cursor() as cur:
            # ① 클릭 좌표를 바운딩박스로 먼저 후보 구역을 빠르게 추림
            cur.execute("""
                SELECT district_id FROM admin_district_bbox
                WHERE min_lng <= %s AND max_lng >= %s
                AND min_lat <= %s AND max_lat >= %s
            """, (lng, lng, lat, lat))
            candidates = [r["district_id"] for r in cur.fetchall()]

            if not candidates:
                conn.close()
                return jsonify({"ok": False, "msg": "행정구역 없음"})

            # ② 후보 구역의 실제 폴리곤 좌표만 가져오기
            fmt = ",".join(["%s"] * len(candidates))
            cur.execute(f"""
                SELECT d.district_id, d.sido, d.sigungu, d.emd,
                g.polygon_index, g.point_order,
                g.longitude, g.latitude
                FROM admin_geometry g
                JOIN admin_district d ON g.district_id = d.district_id
                WHERE g.district_id IN ({fmt})
                ORDER BY g.district_id, g.polygon_index, g.point_order
            """, candidates)
            rows = cur.fetchall()
        conn.close()

        # ③ DB 결과 → 폴리곤 딕셔너리로 조립
        polygons = defaultdict(lambda: {"info": None, "coords": defaultdict(list)})
        for row in rows:
            did = row["district_id"]
            polygons[did]["info"] = (row["sido"], row["sigungu"], row["emd"])
            polygons[did]["coords"][row["polygon_index"]].append(
                (row["longitude"], row["latitude"])
            )

        # ④ 폴리곤 내부에 있는 구역 중 중심이 가장 가까운 것 선택
        best      = None
        best_dist = float('inf')

        for did, data in polygons.items():
            for coords in data["coords"].values():
                if len(coords) < 3:
                    continue    # 삼각형 미만은 폴리곤 불가
                if point_in_polygon(lng, lat, coords):
                    # 폴리곤 중심 계산
                    cx   = sum(c[0] for c in coords) / len(coords)
                    cy   = sum(c[1] for c in coords) / len(coords)
                    dist = (cx - lng) ** 2 + (cy - lat) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        sido, sigungu, emd = data["info"]
                        # None 제거 후 공백으로 연결 (예: "경기도 수원시 영통구")
                        best = " ".join(p for p in [sido, sigungu, emd] if p)

        if best:
            return jsonify({"ok": True, "address": best})
        return jsonify({"ok": False, "msg": "행정구역 없음"})

    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


# ######################################################################
#               날씨 기능  (기상청 API → DB → 웹)
# ######################################################################

# 기상청 API 호출 전용 세션 (일반 requests와 분리)
session_weather = requests.Session()

# 네트워크 불안정 시 자동 재시도 설정
retry_strategy = Retry(
    total=3,          # 최대 3번 재시도
    connect=3,        # 연결 실패 시 3번
    read=3,           # 읽기 실패 시 3번
    backoff_factor=0.5,   # 재시도 간격 (0.5, 1.0, 2.0 초...)
)

adapter = HTTPAdapter(max_retries=retry_strategy)

# https / http 모두 재시도 전략 적용
session_weather.mount("https://", adapter)
session_weather.mount("http://",  adapter)


def to_float(v):
    """API 응답 문자열을 float으로 변환. '-' 나 빈 값은 None 반환."""
    if v is None:
        return None
    v = v.strip()
    if v in ("", "-"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def to_int(v):
    """API 응답 문자열을 int로 변환. '-' 나 빈 값은 None 반환."""
    if v is None:
        return None
    v = v.strip()
    if v in ("", "-"):
        return None
    try:
        return int(v)
    except ValueError:
        return None


def clear_weather():
    """weather_info 테이블 전체 데이터 삭제. 서버 시작/종료 시 호출."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM weather_info")
            print("weather_info 초기화 완료")
    finally:
        conn.close()


def cleanup_weather():
    """6시간 이전 날씨 데이터 삭제. DB 용량 관리 목적."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM weather_info WHERE TM < NOW() - INTERVAL 6 HOUR")
    finally:
        conn.close()


def save_weather_to_db(weather_data: dict):
    """기상 데이터 딕셔너리를 DB에 저장.
    ON DUPLICATE KEY UPDATE: 같은 관측소+시각 데이터가 있으면 UPDATE, 없으면 INSERT."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO weather_info
            (STN, TM, WS, WD, RN, TYPE)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                WS = VALUES(WS),
                WD = VALUES(WD),
                RN = VALUES(RN),
                TYPE = VALUES(TYPE)
            """
            # 관측소별로 저장 (stn = 관측소 코드)
            for stn, data in weather_data.items():
                cur.execute(sql, (stn, data["TM"], data["WS"], data["WD"], data["RN"], data["TYPE"]))
    finally:
        conn.close()


def get_weather_from_db(stn: str):
    """특정 관측소의 가장 최신 날씨 데이터를 DB에서 조회."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT STN, TM, WS, WD, RN, TYPE
                FROM weather_info
                WHERE STN=%s
                ORDER BY TM DESC
                LIMIT 1
            """, (stn,))

            row = cur.fetchone()
            if not row:
                return None

            tm = row["TM"]   # datetime 객체
            # 연/월/일/시를 분리해서 반환 (화면 표시에 활용)
            return {
                "TM":    tm,
                "year":  tm.year,
                "month": tm.month,
                "day":   tm.day,
                "hour":  tm.hour,
                "WS":    row["WS"],   # 풍속 (m/s)
                "WD":    row["WD"],   # 풍향 (deg)
                "RN":    row["RN"],   # 강수량 (mm)
                "TYPE":  row["TYPE"]  # 관측 타입 (GROUND / SEA)
            }
    finally:
        conn.close()


def load_station_data():
    """station 테이블의 전체 관측소 목록을 딕셔너리로 로드.
    서버 시작 시 한 번만 실행되어 메모리에 상주 → nearest_station 함수에서 사용."""
    station_dict = {}
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT STN, STN_NAME, LAT, LON, TYPE FROM station")
            for row in cur.fetchall():
                # 관측소 코드를 키로, 이름/좌표/타입을 값으로 저장
                station_dict[str(row["STN"])] = {
                    "stn_name": row["STN_NAME"],
                    "lat":      float(row["LAT"]),
                    "lon":      float(row["LON"]),
                    "type":     row["TYPE"]   # GROUND(지상) / SEA(해상)
                }
    except Exception as e:
        print("관측소 데이터 로드 오류:", e)
        station_dict = {}
    finally:
        conn.close()
    return station_dict


# 서버 시작 시 관측소 데이터를 메모리에 미리 로드
try:
    station_dict = load_station_data()
    print(f"관측소 데이터 로드: {len(station_dict)}건")
except Exception as e:
    print("관측소 로드 실패:", e)
    station_dict = {}


def nearest_station_wind(lat: float, lon: float):
    """입력 좌표에서 가장 가까운 관측소(풍속 포함) 코드 반환.
    유클리드 거리(제곱합)로 최솟값 탐색 — 정확도보다 속도 우선."""
    min_dist = float("inf")
    nearest  = None

    for stn, info in station_dict.items():
        dist = (lat - info["lat"]) ** 2 + (lon - info["lon"]) ** 2
        if dist < min_dist:
            min_dist = dist
            nearest  = stn

    return nearest


def nearest_station_rain(lat: float, lon: float):
    """강수량 데이터가 있는 지상 관측소 중 가장 가까운 코드 반환.
    해상 부이(SEA)는 강수 데이터 없으므로 GROUND만 필터링."""
    min_dist = float("inf")
    nearest  = None

    for stn, info in station_dict.items():
        if info["type"] != "GROUND":
            continue    # 해상 관측소는 강수 없으므로 건너뜀
        dist = (lat - info["lat"]) ** 2 + (lon - info["lon"]) ** 2
        if dist < min_dist:
            min_dist = dist
            nearest  = stn

    return nearest


def get_ground_weather(tm: str):
    """기상청 지상 관측 API에서 특정 시각의 전국 기상 데이터 수신.
    tm: 'YYYYMMDHH00' 형식 문자열"""
    url    = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
    params = {"tm": tm, "stn": "0", "help": "0", "authKey": API_KEY_WEA}

    try:
        res = session_weather.get(url, params=params, timeout=(5, 15))
        res.raise_for_status()    # HTTP 오류(4xx, 5xx)면 예외 발생
    except Exception as e:
        print("지상 관측 API 오류:", e)
        return {}

    result = {}
    for line in res.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue    # 주석 행 및 빈 행 무시

        data = line.split()

        stn = data[1]   # 관측 지점 코드

        # 관측 일시 파싱
        TM  = datetime.strptime(data[0], "%Y%m%d%H%M")

        # 풍향: 36방위(0~36) → 도(deg, 0~360)로 변환
        WD = to_int(data[2])
        if 0 <= WD <= 36:
            WD = WD * 10
        else:
            WD = None

        # 풍속 (m/s): 음수는 결측 처리
        WS = to_float(data[3])
        if WS < 0:
            WS = None

        # 강수량 (mm): 음수는 결측 처리
        RN = to_float(data[10])
        if RN < 0:
            RN = None

        result[stn] = {
            "TM":   TM,
            "WD":   WD,
            "WS":   WS,
            "RN":   RN,
            "TYPE": "GROUND"
        }

    return result


def get_sea_weather(tm: str):
    """기상청 해상 부이 API에서 해상 기상 데이터 수신.
    해상 부이는 강수 데이터가 없으므로 RN=None."""
    url    = "https://apihub.kma.go.kr/api/typ01/url/kma_buoy.php"
    params = {"tm": tm, "stn": "0", "help": "0", "authKey": API_KEY_WEAS}

    try:
        res = session_weather.get(url, params=params, timeout=(5, 15))
        res.raise_for_status()
    except Exception as e:
        print("해상 관측 API 오류:", e)
        return {}

    result = {}
    for line in res.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        data = line.split()
        stn  = data[1]
        TM   = datetime.strptime(data[0], "%Y%m%d%H%M")

        # 해상 부이는 풍향을 바로 deg(0~360)로 제공
        WD = to_int(data[2])
        if not (0 <= WD <= 360):
            WD = None

        WS = to_float(data[3])
        if WS < 0:
            WS = None

        result[stn] = {
            "TM":   TM,
            "WD":   WD,
            "WS":   WS,
            "RN":   None,      # 해상 부이는 강수 미측정
            "TYPE": "SEA"
        }

    return result


def update_weather():
    """현재 시각의 지상+해상 기상 데이터를 API에서 받아 DB에 저장.
    성공 시 True, 실패 시 False 반환 → 스케줄러에서 재시도 여부 결정."""
    try:
        now    = datetime.now()
        tm_now = now.strftime("%Y%m%d%H00")

        ground_now = get_ground_weather(tm_now)
        time.sleep(1)    # API 과부하 방지 1초 대기
        sea_now    = get_sea_weather(tm_now)

        # 지상 + 해상 데이터 합치기 (같은 관측소 코드면 나중 것으로 덮어씀)
        weather = {**ground_now, **sea_now}

        if not weather:
            print("날씨 데이터 없음")
            return False

        save_weather_to_db(weather)
        cleanup_weather()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 날씨 업데이트 완료 : {len(weather)}건")
        return True

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 날씨 업데이트 오류: {e}")
        return False


def initial_weather_load():
    """서버 시작 시 현재 시각 + 1시간 전 데이터 2회 호출하여 DB를 채움.
    처음 시작 시 DB가 비어있어서 날씨 조회가 안 되는 문제를 해결."""
    now     = datetime.now()
    tm_now  = now.strftime("%Y%m%d%H00")
    tm_prev = (now - timedelta(hours=1)).strftime("%Y%m%d%H00")

    # 이전 시각 데이터 수집
    ground_prev = get_ground_weather(tm_prev)
    sea_prev    = get_sea_weather(tm_prev)
    time.sleep(1)

    # 현재 시각 데이터 수집
    ground_now = get_ground_weather(tm_now)
    sea_now    = get_sea_weather(tm_now)
    time.sleep(1)

    # 4개 딕셔너리를 합침 (현재 시각이 이전 시각을 덮어씀 → 최신 데이터 우선)
    weather = {**ground_prev, **sea_prev, **ground_now, **sea_now}
    save_weather_to_db(weather)
    print(f"초기 데이터 저장 : {len(weather)}건")


def weather_scheduler():
    """매 정각+2분에 날씨 데이터를 자동 업데이트하는 스케줄러.
    별도 스레드(daemon=True)로 실행되며 서버가 종료될 때 같이 종료됨.
    업데이트 실패 시 1분 간격으로 최대 5회 재시도."""
    while True:
        now      = datetime.now()
        # 다음 정각+2분 계산 (예: 현재 14:35 → 다음 실행 15:02)
        next_run = now.replace(minute=2, second=0, microsecond=0)
        if now.minute >= 2:
            next_run += timedelta(hours=1)

        wait_seconds = (next_run - now).total_seconds()
        print(f"[{now.strftime('%H:%M:%S')}] 다음 정규 업데이트: {next_run.strftime('%H:%M:%S')}")
        time.sleep(wait_seconds)

        # 기본 업데이트 시도
        success = update_weather()

        # 실패 시 최대 5회, 1분 간격 재시도
        retry_count = 0
        while not success and retry_count < 5:
            retry_count += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 업데이트 실패, 재시도 {retry_count}/5 (1분 후)")
            time.sleep(60)
            success = update_weather()


def start_weather_thread():
    """날씨 스케줄러를 백그라운드 데몬 스레드로 시작."""
    t = threading.Thread(target=weather_scheduler, daemon=True)
    t.start()
    print("기상 정보 업데이트 스레드 시작")


def get_sun_time(lat: float, lon: float):
    """astral 라이브러리로 입력 좌표의 오늘 일출/일몰 시각을 계산.
    한국 시간대(Asia/Seoul) 기준으로 반환."""
    city = LocationInfo(latitude=lat, longitude=lon, timezone="Asia/Seoul")
    s    = sun(city.observer, date=datetime.now().date(),
               tzinfo=pytz.timezone("Asia/Seoul"))
    return s["sunrise"].time(), s["sunset"].time()


def flight_status(sunrise, sunset, wind: float, rain: float):
    """풍속, 강수, 야간 여부로 비행 가능 여부를 판단.
    반환: (상태 문자열, 이유 문자열)
    상태: '비행 가능' / '비행 주의' / '비행 위험' / '비행 금지' / '비행 불가'"""
    now = datetime.now().time()

    # 데이터 없으면 판단 불가
    if (wind is None and rain is None) or \
       (wind is None and rain == 0.0) or \
       (rain is None and wind == 0.0):
        return "비행 판단 불가", "기상 데이터를 불러올 수 없습니다."

    # 야간(일몰 후 ~ 일출 전)이면 비행 불가
    if not (sunrise <= now <= sunset):
        return "비행 불가", "야간 비행 불가(일몰 후 ~ 일출 전)"

    # 강수 시 비행 위험
    if rain is not None and rain > 0:
        return "비행 위험", f"강수({rain}mm) 상황으로 비행 불가"

    # 풍속별 단계적 판단
    if wind is not None and wind >= 10:
        return "비행 금지",  f"강풍({wind}m/s) 상황으로 비행 불가"
    if wind is not None and wind >= 8:
        return "비행 위험",  f"강풍({wind}m/s) 상황으로 비행 위험"
    if wind is not None and wind >= 6:
        return "비행 주의",  f"강풍({wind}m/s) 상황으로 비행 주의"
    if wind is not None and wind >= 4:
        return "비행 가능",  f"다소 바람이 있어 비행 주의 ({wind}m/s)"

    return "비행 가능", "안전한 비행 조건입니다."


@app.route("/api/weather")
def api_weather():
    """위경도를 받아 가장 가까운 관측소의 기상 데이터 + 비행 가능 여부를 JSON으로 반환."""
    try:
        lat = float(request.args.get("lat", 37.5665))    # 기본값: 서울 시청
        lon = float(request.args.get("lon", 126.9780))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "잘못된 좌표"})

    # 일출/일몰 계산
    sunrise, sunset = get_sun_time(lat, lon)

    # 가장 가까운 관측소 탐색
    stn_rain = nearest_station_rain(lat, lon) or "108"   # 강수 전용 (지상만)
    stn_wind = nearest_station_wind(lat, lon) or "108"   # 풍속 전용 (지상+해상)
    stn_name = station_dict.get(stn_wind, {}).get("stn_name", "Unknown")

    # DB에서 최신 날씨 가져오기
    weather_rain = get_weather_from_db(stn_rain)
    weather_wind = get_weather_from_db(stn_wind)

    if not weather_rain or not weather_wind:
        return jsonify({"ok": False, "msg": "해당 지역 기상 데이터 조회 불가"})

    # 비행 가능 여부 판단 (날씨 기반)
    status, reason = flight_status(
        sunrise, sunset,
        weather_wind.get("WS") or weather_wind.get("wind"),
        weather_rain.get("RN") or weather_rain.get("rain")
    )

    return jsonify({
        "ok":       True,
        "lat":      lat,
        "lon":      lon,
        "stn_name": stn_name,
        "year":     weather_rain.get("year"),
        "month":    weather_rain.get("month"),
        "day":      weather_rain.get("day"),
        "hour":     weather_rain.get("hour"),
        "wind_dir": weather_wind.get("WD") or weather_wind.get("wind_dir"),
        "wind":     weather_wind.get("WS") or weather_wind.get("wind"),
        "rain":     weather_rain.get("RN") or weather_rain.get("rain"),
        "sunrise":  sunrise.strftime("%H:%M"),
        "sunset":   sunset.strftime("%H:%M"),
        "status":   status,
        "reason":   reason
    })


def shutdown_cleanup():
    """서버 종료 시 weather_info 테이블을 비움.
    atexit.register()에 등록되어 있어 Ctrl+C 등으로 종료 시 자동 호출."""
    print("서버 종료 → weather_info clear")
    clear_weather()


# 서버 종료 시 shutdown_cleanup 자동 실행 등록
atexit.register(shutdown_cleanup)


# ######################################################################
#              사전 확인 API  /api/prechecks
#  비행 전 체크리스트 항목을 DB에서 조회해 JSON으로 반환
# ######################################################################
@app.route("/api/prechecks")
def api_prechecks():
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            # sort_order 순서대로 체크리스트 항목 조회
            cur.execute("""
                SELECT check_id, icon, content
                FROM pre_check
                ORDER BY sort_order, check_id
            """)
            rows = cur.fetchall()
        conn.close()
        # 필요한 필드만 골라서 JSON 배열로 반환
        return jsonify({"ok": True, "items": [
            {"id": r["check_id"], "icon": r["icon"], "content": r["content"]}
            for r in rows
        ]})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


# ######################################################################
#              비행 허가 신청 API  /api/permit/submit
#  FormData 방식(파일 업로드 포함) POST 요청 처리
# ######################################################################
@app.route("/api/permit/submit", methods=["POST"])
def api_permit_submit():
    if "user_id" not in session:
        return jsonify({"ok": False, "msg": "로그인이 필요합니다."})
    try:
        user_id         = session["user_id"]
        purpose         = request.form.get("purpose", "").strip()         # 비행 목적
        drone_type      = request.form.get("drone_type", "").strip()      # 기체 종류
        start_date      = request.form.get("start_date", "").strip()      # 시작 날짜
        end_date        = request.form.get("end_date", "").strip()        # 종료 날짜
        start_time      = request.form.get("start_time", "").strip() or None
        end_time        = request.form.get("end_time",   "").strip() or None
        latitude        = request.form.get("latitude")                    # 위도
        longitude       = request.form.get("longitude")                   # 경도
        radius          = request.form.get("radius", 500)                 # 비행 반경 (m)
        flight_altitude = request.form.get("flight_altitude", None)       # 비행 고도 (m)

        # 필수 항목 누락 체크
        if not all([purpose, start_date, end_date]):
            return jsonify({"ok": False, "msg": "필수 항목(사용 목적, 비행 기간)을 모두 입력해주세요."})

        # 타입 변환
        lat = float(latitude)  if latitude  else None
        lng = float(longitude) if longitude else None
        r   = int(radius)      if radius    else 500
        alt = int(flight_altitude) if flight_altitude else None

        conn = get_conn()
        with conn.cursor() as cur:
            # 촬영 신청 여부 (1이면 촬영 신청함)
            photo_req = 1 if request.form.get('photo_request', '0') == '1' else 0
            try:
                # 모든 컬럼이 있는 경우 (flight_altitude, start_time 등 포함)
                cur.execute("""
                    INSERT INTO flight_request
                        (user_id, purpose, drone_type, start_date, end_date,
                         start_time, end_time, latitude, longitude, radius,
                         flight_altitude, photo_request, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'대기')
                """, (user_id, purpose, drone_type, start_date, end_date,
                      start_time, end_time, lat, lng, r, alt, photo_req))
            except Exception:
                # 일부 컬럼이 DB에 없는 구버전 스키마 호환 처리
                cur.execute("""
                    INSERT INTO flight_request
                        (user_id, purpose, drone_type, start_date, end_date,
                         latitude, longitude, radius, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'대기')
                """, (user_id, purpose, drone_type, start_date, end_date, lat, lng, r))
            request_id = cur.lastrowid    # 방금 저장된 신청서 ID

            # 첨부 파일 저장 처리
            files = request.files.getlist("attachments")
            for f in files:
                if f and f.filename and allowed_file(f.filename):
                    safe  = secure_filename(f.filename)         # 안전한 파일명으로 변환
                    saved = f"{request_id}_{safe}"              # 신청ID_파일명으로 저장
                    f.save(os.path.join(UPLOAD_FOLDER, saved))
                    cur.execute("""
                        INSERT INTO permit_files (request_id, original_name, saved_name, file_size)
                        VALUES (%s, %s, %s, %s)
                    """, (request_id, f.filename, saved, len(f.read()) if hasattr(f, 'read') else 0))
        conn.close()
        return jsonify({"ok": True, "msg": "신청이 완료되었습니다.", "request_id": request_id})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/my_permits/<int:req_id>/update", methods=["POST"])
def api_permit_update(req_id):
    """허가 신청 수정 API. '대기' 상태인 신청만 수정 가능."""
    if "user_id" not in session:
        return jsonify({"ok": False, "msg": "로그인이 필요합니다."})
    try:
        data = request.get_json()
        purpose         = data.get("purpose", "").strip()
        drone_type      = data.get("drone_type", "").strip()
        start_date      = data.get("start_date", "").strip()
        end_date        = data.get("end_date", "").strip()
        latitude        = data.get("latitude")
        longitude       = data.get("longitude")
        radius          = data.get("radius", 500)
        flight_altitude = data.get("flight_altitude", None)

        if not all([purpose, start_date, end_date]):
            return jsonify({"ok": False, "msg": "필수 항목을 입력해주세요."})

        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE flight_request
                SET purpose=%s, drone_type=%s, start_date=%s, end_date=%s,
                    latitude=%s, longitude=%s, radius=%s, flight_altitude=%s
                WHERE request_id=%s AND user_id=%s AND status='대기'
            """, (purpose, drone_type, start_date, end_date,
                  latitude, longitude, radius, flight_altitude,
                  req_id, session["user_id"]))
            # rowcount == 0 이면 해당 신청이 없거나 '대기' 상태가 아닌 것
            if cur.rowcount == 0:
                conn.close()
                return jsonify({"ok": False, "msg": "수정할 수 없습니다. 대기 중인 신청만 수정 가능합니다."})
        conn.close()
        return jsonify({"ok": True, "msg": "수정이 완료되었습니다."})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/my_permits/<int:req_id>/files")
def api_permit_files(req_id):
    """특정 허가 신청의 첨부파일 목록 반환. 본인 또는 관리자만 조회 가능."""
    is_admin = session.get("is_admin", False)
    user_id  = session.get("user_id")
    if not is_admin and not user_id:
        return jsonify({"ok": False, "msg": "로그인이 필요합니다."})
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            if not is_admin:
                # 일반 유저는 본인 신청만 확인 가능
                cur.execute(
                    "SELECT request_id FROM flight_request WHERE request_id=%s AND user_id=%s",
                    (req_id, user_id)
                )
                if not cur.fetchone():
                    conn.close()
                    return jsonify({"ok": False, "msg": "권한이 없습니다."})
            cur.execute("""
                SELECT file_id, original_name, saved_name, file_size, uploaded_at
                FROM permit_files WHERE request_id=%s ORDER BY file_id
            """, (req_id,))
            files = [{
                "file_id":       r["file_id"],
                "original_name": r["original_name"],
                "saved_name":    r["saved_name"],
                "file_size":     r["file_size"],
                "download_url":  f"/api/permit/download/{r['saved_name']}"
            } for r in cur.fetchall()]
        conn.close()
        return jsonify({"ok": True, "files": files})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/permit/download/<path:filename>")
def api_permit_download(filename):
    """첨부파일 다운로드. 관리자 또는 본인만 가능."""
    is_admin = session.get("is_admin", False)
    user_id  = session.get("user_id")
    if not is_admin and not user_id:
        return jsonify({"ok": False, "msg": "로그인이 필요합니다."}), 403
    try:
        from flask import send_from_directory
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception:
        return jsonify({"ok": False, "msg": "파일을 찾을 수 없습니다."}), 404


@app.route("/api/my_permits")
def api_my_permits():
    """로그인 유저의 전체 허가 신청 목록 반환 (지도 표시 + 상태 패널용)."""
    if "user_id" not in session:
        return jsonify({"ok": True, "permits": []})   # 비로그인이면 빈 배열 반환
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT request_id, purpose, start_date, end_date,
                       COALESCE(start_time,'') AS start_time,
                       COALESCE(end_time,'') AS end_time,
                       latitude, longitude, radius,
                       COALESCE(drone_type,'') AS drone_type,
                       COALESCE(flight_altitude,0) AS flight_altitude,
                       COALESCE(photo_request,0) AS photo_request,
                       status,
                       COALESCE(reject_reason, '') AS reject_reason
                FROM flight_request
                WHERE user_id = %s
                  AND latitude  IS NOT NULL
                  AND longitude IS NOT NULL
                ORDER BY request_id DESC
            """, (session["user_id"],))
            rows = cur.fetchall()
        conn.close()
        # DB 결과를 JS에서 쓰기 편한 형태로 변환
        permits = [{
            "id":              r["request_id"],
            "purpose":         r["purpose"],
            "start_date":      str(r["start_date"]),
            "end_date":        str(r["end_date"]),
            "start_time":      r["start_time"],
            "end_time":        r["end_time"],
            "lat":             float(r["latitude"]),
            "lng":             float(r["longitude"]),
            "radius":          r["radius"] or 500,
            "drone_type":      r["drone_type"],
            "flight_altitude": r["flight_altitude"],
            "status":          r["status"],
            "reject_reason":   r.get("reject_reason", "") or "",
            "photo_request":   int(r.get("photo_request", 0) or 0)
        } for r in rows]
        return jsonify({"ok": True, "permits": permits})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/all_permits")
def api_all_permits():
    """관리자 전용 — 모든 유저의 허가 신청 목록 (지도 표시용).
    날짜 범위 필터 적용 가능 (?date_from=2025-01-01&date_to=2025-01-31)."""
    if not session.get("is_admin"):
        return jsonify({"ok": False, "msg": "관리자 권한 필요"})
    try:
        date_from = request.args.get("date_from", "").strip()
        date_to   = request.args.get("date_to",   "").strip()

        # 날짜 범위 WHERE 절 동적 생성
        where_extra = ""
        params = []
        if date_from and date_to:
            where_extra = "AND fr.start_date <= %s AND fr.end_date >= %s"
            params = [date_to, date_from]
        elif date_from:
            where_extra = "AND fr.end_date >= %s"
            params = [date_from]
        elif date_to:
            where_extra = "AND fr.start_date <= %s"
            params = [date_to]

        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT fr.request_id, u.name AS user_name, fr.purpose,
                       fr.start_date, fr.end_date,
                       fr.latitude, fr.longitude,
                       COALESCE(fr.radius, 500) AS radius,
                       fr.status
                FROM flight_request fr
                JOIN users u ON fr.user_id = u.user_id
                WHERE fr.latitude  IS NOT NULL
                  AND fr.longitude IS NOT NULL
                  {where_extra}
                ORDER BY fr.request_id DESC
            """, params)
            rows = cur.fetchall()
        conn.close()

        # 한글 상태 → 영어 변환 (JS colorMap 키와 맞춤)
        status_map = {'승인': 'approved', '대기': 'pending', '거절': 'rejected'}
        permits = [{
            "id":         r["request_id"],
            "user_name":  r["user_name"],
            "purpose":    r["purpose"],
            "start_date": str(r["start_date"]),
            "end_date":   str(r["end_date"]),
            "lat":    float(r["latitude"]),
            "lng":    float(r["longitude"]),
            "radius": r["radius"] or 500,
            "status": status_map.get(r["status"], "pending")
        } for r in rows]
        return jsonify({"ok": True, "permits": permits})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/notices")
def api_notices():
    """최신 공지사항 10건을 JSON으로 반환. 메인 화면 공지 탭에서 사용."""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT notice_id, title, content, created_at
                FROM notice
                ORDER BY notice_id DESC
                LIMIT 10
            """)
            rows = cur.fetchall()
        conn.close()
        notices = []
        for r in rows:
            notices.append({
                "id":         r["notice_id"],
                "title":      r["title"],
                "content":    r["content"] or "",
                # created_at이 None이면 '-' 출력
                "created_at": r["created_at"].strftime("%Y.%m.%d") if r["created_at"] else "-"
            })
        return jsonify({"ok": True, "notices": notices})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


# ######################################################################
#                         서버 실행 진입점
# ######################################################################
if __name__ == "__main__":
    clear_weather()            # 서버 재시작 시 오래된 날씨 데이터 초기화
    initial_weather_load()     # 현재 + 1시간 전 날씨 데이터 미리 로드

    # 날씨 자동 업데이트 스레드 시작 (백그라운드 데몬)
    t = threading.Thread(target=weather_scheduler)
    t.daemon = True
    t.start()

    # Flask 개발 서버 실행 (모든 IP에서 5000번 포트로 접속 가능)
    app.run(host="0.0.0.0", port=5000, debug=True)
