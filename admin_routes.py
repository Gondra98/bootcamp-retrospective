# ================================================================
#  admin_routes.py  —  관리자 전용 라우트 (Blueprint)
#  Flask Blueprint를 이용해 /admin/* URL을 별도 파일로 분리.
#  app.py가 너무 길어지지 않도록 관리자 기능만 여기서 관리.
# ================================================================

# Flask에서 필요한 기능들
# Blueprint    - 라우트를 별도 파일로 분리하는 모듈
# render_template - HTML 템플릿 렌더링
# request      - 클라이언트 요청 데이터
# redirect     - 다른 URL로 이동
# url_for      - 함수명으로 URL 생성
# flash        - 1회성 메시지
# session      - 로그인 세션 정보
from flask import Blueprint, render_template, request, redirect, url_for, flash, session

# functools.wraps: 데코레이터 만들 때 원본 함수 이름/설명 보존
from functools import wraps

# MariaDB 접속 드라이버
import pymysql

# 환경변수 읽기
import os

# 날짜 계산 (오늘 날짜, +N일 등)
from datetime import date, timedelta

# Blueprint 인스턴스 생성
# 'admin' - 블루프린트 이름 (url_for('admin.함수명') 형태로 사용)
# url_prefix='/admin' - 이 블루프린트의 모든 URL 앞에 /admin이 붙음
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# 페이지당 목록 표시 건수
PER_PAGE = 10


# ── DB 연결 함수 ─────────────────────────────────────────────
def get_conn():
    """관리자 페이지 전용 DB 연결 함수.
    app.py의 get_conn()과 동일하지만 Blueprint는 별도 파일이라 중복 정의."""
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),    # 기본 포트 3306
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",                        # 한글 + 이모지 지원
        cursorclass=pymysql.cursors.DictCursor,   # 결과를 dict로 반환
        autocommit=True,                          # 자동 커밋
    )


# ── 관리자 권한 체크 데코레이터 ──────────────────────────────
def admin_required(f):
    """@admin_required 데코레이터 — 관리자 세션이 없으면 로그인 페이지로 강제 이동.
    수업에서 배운 데코레이터 패턴 활용: @wraps(f)로 원본 함수 정보 유지."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # session['is_admin']이 True가 아니면 접근 차단
        if not session.get('is_admin'):
            flash('관리자만 접근할 수 있습니다.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)    # 정상이면 원래 함수 실행
    return decorated


def get_pending_count(cur):
    """'대기' 상태인 허가 신청 건수 조회. 사이드바 뱃지 숫자에 표시됨."""
    cur.execute("SELECT COUNT(*) AS cnt FROM flight_request WHERE status = '대기'")
    return cur.fetchone()['cnt']


def to_map_xy(lat, lng):
    """위경도 좌표를 관리자 지도 SVG 상의 % 위치로 변환.
    한국 전체 범위(위도 33~38.7, 경도 125~130)를 0~100% 좌표계에 매핑.
    예외 발생 시 중앙(50, 50)으로 fallback."""
    LAT_MIN, LAT_MAX = 33.0, 38.7
    LNG_MIN, LNG_MAX = 125.0, 130.0
    try:
        # X(%): 경도 기준, 좌→우 (5~95% 범위 사용)
        x = round((float(lng) - LNG_MIN) / (LNG_MAX - LNG_MIN) * 90 + 5, 2)
        # Y(%): 위도 기준, 위→아래 (10~90% 범위 사용, 위도는 위가 높으므로 1-로 반전)
        y = round((1 - (float(lat) - LAT_MIN) / (LAT_MAX - LAT_MIN)) * 80 + 10, 2)
        return x, y
    except Exception:
        return 50, 50    # 계산 불가 시 지도 중앙


# =====================================================================
#  관리자 대시보드  GET /admin/
# =====================================================================
@admin_bp.route('/')
@admin_required   # 관리자 권한 없으면 접근 불가
def admin_dashboard():
    """관리자 메인 화면 — 이번 달 통계, 최근 신청 5건, 최근 가입 5명을 표시."""
    conn = get_conn()
    cur  = conn.cursor()

    # 이번 달 허가 신청 통계 (전체/대기/승인/거절)
    cur.execute("""
        SELECT
            COUNT(*)                   AS total,     -- 전체 건수
            SUM(status = '대기')       AS pending,   -- 대기 건수
            SUM(status = '승인')       AS approved,  -- 승인 건수
            SUM(status = '거절')       AS rejected   -- 거절 건수
        FROM flight_request
        WHERE MONTH(start_date) = MONTH(CURDATE())   -- 이번 달
          AND YEAR(start_date)  = YEAR(CURDATE())    -- 올해
    """)
    stats = cur.fetchone()

    # 최근 허가 신청 5건 (신청자 이름 JOIN)
    cur.execute("""
        SELECT
            fr.request_id,
            u.name      AS applicant_name,
            fr.purpose,
            fr.status,
            fr.start_date AS created_at
        FROM flight_request fr
        JOIN users u ON fr.user_id = u.user_id
        ORDER BY fr.request_id DESC
        LIMIT 5
    """)
    raw_recent = cur.fetchall()

    # 한글 상태값 → 영어 변환 (HTML 템플릿에서 badge CSS 클래스명으로 사용)
    status_map_to_en = {'대기': 'pending', '승인': 'approved', '거절': 'rejected'}
    recent_permits = []
    for r in raw_recent:
        r['status'] = status_map_to_en.get(r['status'], r['status'])
        recent_permits.append(r)

    # 최근 가입 회원 5명
    cur.execute("""
        SELECT login_id AS user_id, name, created_at
        FROM users
        ORDER BY created_at DESC
        LIMIT 5
    """)
    recent_users = cur.fetchall()

    pending_count = get_pending_count(cur)    # 사이드바 뱃지용 대기 건수
    conn.close()

    return render_template('admin.html',
        active_tab='dashboard',           # 현재 활성 탭 표시
        pending_count=pending_count,      # 사이드바 뱃지
        stats=stats,                      # 이번 달 통계
        recent_permits=recent_permits,    # 최근 신청 5건
        recent_users=recent_users,        # 최근 가입 5명
    )


# =====================================================================
#  비행 허가 신청 목록  GET /admin/approval
#  검색/필터/페이지네이션 지원
# =====================================================================
@admin_bp.route('/approval')
@admin_required
def admin_approval():
    """허가 신청 목록 — 검색어(q), 상태 필터(status), 페이지(page) 파라미터 지원."""
    q      = request.args.get('q', '').strip()         # 검색어 (이름 or 목적)
    status = request.args.get('status', '')            # 상태 필터 (영어)
    page   = int(request.args.get('page', 1))          # 현재 페이지 (기본 1)
    offset = (page - 1) * PER_PAGE                     # SQL OFFSET 계산

    # 화면은 영어(pending/approved/rejected), DB는 한글(대기/승인/거절)
    status_map_to_kr = {'pending': '대기', 'approved': '승인', 'rejected': '거절'}

    # WHERE 조건 동적 생성
    conditions, params = [], []
    if q:
        # 신청자 이름 또는 비행 목적에서 검색 (LIKE %검색어%)
        conditions.append("(u.name LIKE %s OR fr.purpose LIKE %s)")
        params += [f'%{q}%', f'%{q}%']
    if status and status in status_map_to_kr:
        conditions.append("fr.status = %s")
        params.append(status_map_to_kr[status])    # 영어 → 한글 변환 후 DB 조회

    where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

    conn = get_conn()
    cur  = conn.cursor()

    # 전체 건수 조회 (페이지네이션 계산용)
    cur.execute(f"""
        SELECT COUNT(*) AS cnt
        FROM flight_request fr
        JOIN users u ON fr.user_id = u.user_id
        {where}
    """, params)
    total_count = cur.fetchone()['cnt']
    total_pages = max(1, -(-total_count // PER_PAGE))    # 올림 나눗셈 (총 페이지 수)

    # 실제 목록 조회 (한 페이지 분량)
    cur.execute(f"""
        SELECT
            fr.request_id               AS permit_id,
            u.name                      AS applicant_name,
            u.login_id                  AS applicant_id,
            u.phone                     AS applicant_phone,
            fr.purpose                  AS flight_area,
            fr.purpose,
            fr.start_date,
            fr.end_date,
            COALESCE(fr.start_time,'')  AS start_time,
            COALESCE(fr.end_time,'')    AS end_time,
            fr.drone_type,
            COALESCE(fr.flight_altitude,0) AS flight_altitude,
            fr.status,
            fr.latitude,
            fr.longitude,
            COALESCE(fr.radius,500)     AS radius,
            COALESCE(fr.photo_request,0) AS photo_request,
            fr.reject_reason
        FROM flight_request fr
        JOIN users u ON fr.user_id = u.user_id
        {where}
        ORDER BY fr.request_id DESC
        LIMIT %s OFFSET %s
    """, params + [PER_PAGE, offset])
    raw_permits = cur.fetchall()

    # 상태 한글 → 영어 변환
    status_map_to_en = {'대기': 'pending', '승인': 'approved', '거절': 'rejected'}
    permits = []
    for p in raw_permits:
        p['status'] = status_map_to_en.get(p['status'], p['status'])
        permits.append(p)

    pending_count = get_pending_count(cur)
    conn.close()

    return render_template('admin.html',
        active_tab='approval',
        pending_count=pending_count,
        permits=permits,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
    )


# ── 개별 허가 승인  GET /admin/approval/<id>/approve ────────────
@admin_bp.route('/approval/<int:permit_id>/approve')
@admin_required
def admin_permit_approve(permit_id):
    """허가 신청 1건 승인 처리. status를 '승인'으로 UPDATE."""
    admin_id = session.get('admin_id', 1)    # 처리한 관리자 ID 기록
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "UPDATE flight_request SET status = '승인', admin_id = %s WHERE request_id = %s",
        (admin_id, permit_id)
    )
    conn.commit()
    conn.close()
    flash('✅ 허가가 승인되었습니다.', 'success')
    return redirect(url_for('admin.admin_approval'))


# ── 개별 허가 거절  POST /admin/approval/<id>/reject ─────────────
@admin_bp.route('/approval/<int:permit_id>/reject', methods=['POST'])
@admin_required
def admin_permit_reject(permit_id):
    """허가 신청 1건 거절 처리. 거절 사유도 함께 저장."""
    reason   = request.form.get('reason', '').strip()   # 거절 사유 (폼에서 입력)
    admin_id = session.get('admin_id', 1)
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "UPDATE flight_request SET status = '거절', admin_id = %s, reject_reason = %s WHERE request_id = %s",
        (admin_id, reason or None, permit_id)    # 사유 없으면 None
    )
    conn.commit()
    conn.close()
    flash('❌ 허가가 거절되었습니다.', 'error')
    return redirect(url_for('admin.admin_approval'))


# ── 일괄 처리  POST /admin/approval/bulk ─────────────────────────
@admin_bp.route('/approval/bulk', methods=['POST'])
@admin_required
def admin_approval_bulk():
    """체크박스로 선택한 여러 신청을 한 번에 승인/거절/삭제."""
    action     = request.form.get('action')             # 'approve' / 'reject' / 'delete'
    permit_ids = request.form.getlist('permit_ids')     # 선택된 신청 ID 목록

    if not permit_ids:
        flash('선택된 항목이 없습니다.', 'error')
        return redirect(url_for('admin.admin_approval'))
    if action not in ('approve', 'reject', 'delete'):
        flash('잘못된 요청입니다.', 'error')
        return redirect(url_for('admin.admin_approval'))

    conn = get_conn()
    cur  = conn.cursor()
    # IN 절에 들어갈 %s 자리표시자를 선택 건수만큼 생성
    placeholders = ','.join(['%s'] * len(permit_ids))

    if action == 'delete':
        cur.execute(
            f"DELETE FROM flight_request WHERE request_id IN ({placeholders})",
            permit_ids
        )
        conn.commit()
        conn.close()
        flash(f'{len(permit_ids)}건이 삭제되었습니다.', 'info')
        return redirect(url_for('admin.admin_approval'))

    # 승인 또는 거절로 일괄 상태 변경
    status   = '승인' if action == 'approve' else '거절'
    admin_id = session.get('admin_id', 1)

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"UPDATE flight_request SET status = %s, admin_id = %s WHERE request_id IN ({placeholders})",
        [status, admin_id] + permit_ids
    )
    conn.close()

    label = '승인' if action == 'approve' else '거절'
    flash(f'{len(permit_ids)}건이 일괄 {label} 처리되었습니다.', 'success')
    return redirect(url_for('admin.admin_approval'))


# =====================================================================
#  허가 지역 지도  GET /admin/map
#  날짜 범위 필터로 허가 신청 구역을 지도에 표시
# =====================================================================
@admin_bp.route('/map')
@admin_required
def admin_map():
    """관리자 지도 화면 — 날짜 범위 내 허가 신청 구역을 Vworld 지도에 표시."""
    date_from = request.args.get('date_from', '').strip()
    date_to   = request.args.get('date_to',   '').strip()

    # 날짜 파라미터 없으면 오늘 ~ 오늘+7일로 기본값 설정
    today = date.today()
    if not date_from and not date_to:
        date_from = today.strftime('%Y-%m-%d')
        date_to   = (today + timedelta(days=7)).strftime('%Y-%m-%d')

    conn = get_conn()
    cur  = conn.cursor()

    # 날짜 범위 WHERE 조건 동적 생성
    where  = ""
    params = []
    if date_from and date_to:
        # 신청 기간이 필터 범위와 겹치는 건만 조회
        where  = "WHERE fr.start_date <= %s AND fr.end_date >= %s"
        params = [date_to, date_from]
    elif date_from:
        where  = "WHERE fr.end_date >= %s"
        params = [date_from]
    elif date_to:
        where  = "WHERE fr.start_date <= %s"
        params = [date_to]

    cur.execute(f"""
        SELECT
            fr.request_id   AS permit_id,
            u.name          AS applicant_name,
            fr.purpose      AS flight_area,
            fr.status,
            fr.latitude,
            fr.longitude,
            COALESCE(fr.radius, 500) AS radius,
            fr.start_date,
            fr.end_date
        FROM flight_request fr
        JOIN users u ON fr.user_id = u.user_id
        {where}
        ORDER BY fr.request_id DESC
    """, params)
    raw = cur.fetchall()

    # 상태 변환 + 지도 SVG 좌표(%) 계산
    status_map_to_en = {'대기': 'pending', '승인': 'approved', '거절': 'rejected'}
    map_permits = []
    for p in raw:
        p['status']            = status_map_to_en.get(p['status'], p['status'])
        p['map_x'], p['map_y'] = to_map_xy(p['latitude'], p['longitude'])
        map_permits.append(p)

    pending_count = get_pending_count(cur)
    conn.close()

    return render_template('admin.html',
        active_tab='map',
        pending_count=pending_count,
        map_permits=map_permits,
        api_key=os.getenv("VWORLD_API_KEY", ""),    # 지도 API 키
        date_from=date_from,
        date_to=date_to,
    )


# =====================================================================
#  회원 정보 조회  GET /admin/members
#  검색/드론 등록 여부/가입 상태 필터 + 페이지네이션
# =====================================================================
@admin_bp.route('/members')
@admin_required
def admin_members():
    """회원 목록 — 가입 대기 회원이 상단에 표시되도록 정렬."""
    q           = request.args.get('q', '').strip()              # 이름 or 아이디 검색
    drone       = request.args.get('drone', '')                  # 드론 등록 여부 필터
    user_status = request.args.get('user_status', '')            # 가입 승인 상태 필터
    page        = int(request.args.get('page', 1))
    offset      = (page - 1) * PER_PAGE

    # 동적 WHERE 조건 생성
    conditions, params = [], []
    if q:
        conditions.append("(u.login_id LIKE %s OR u.name LIKE %s)")
        params += [f'%{q}%', f'%{q}%']
    if drone == 'registered':
        conditions.append("d.drone_id IS NOT NULL")     # 드론 등록한 회원
    elif drone == 'unregistered':
        conditions.append("d.drone_id IS NULL")         # 드론 미등록 회원
    if user_status:
        conditions.append("u.status = %s")
        params.append(user_status)

    where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

    conn = get_conn()
    cur  = conn.cursor()

    # 가입 대기 회원 수 (사이드바 뱃지용)
    cur.execute("SELECT COUNT(*) AS cnt FROM users WHERE status = '대기'")
    member_pending_count = cur.fetchone()['cnt']

    # 전체 회원 수 (페이지네이션 계산)
    cur.execute(f"""
        SELECT COUNT(*) AS cnt
        FROM users u
        LEFT JOIN drone d ON u.user_id = d.user_id
        {where}
    """, params)
    total_users = cur.fetchone()['cnt']
    total_pages = max(1, -(-total_users // PER_PAGE))

    # 회원 목록 조회
    # CASE WHEN: 대기 회원을 맨 위로 정렬 (0 = 대기, 1 = 나머지)
    cur.execute(f"""
        SELECT
            u.user_id,
            u.login_id,
            u.name,
            u.birth                  AS birth_date,
            u.phone,
            u.status,
            u.created_at,
            (d.drone_id IS NOT NULL) AS has_drone    -- 드론 등록 여부 (0 or 1)
        FROM users u
        LEFT JOIN drone d ON u.user_id = d.user_id
        {where}
        ORDER BY
            CASE u.status WHEN '대기' THEN 0 ELSE 1 END,  -- 대기 먼저
            u.created_at DESC
        LIMIT %s OFFSET %s
    """, params + [PER_PAGE, offset])
    users = cur.fetchall()

    pending_count = get_pending_count(cur)
    conn.close()

    return render_template('admin.html',
        active_tab='members',
        pending_count=pending_count,
        member_pending_count=member_pending_count,
        users=users,
        total_users=total_users,
        total_pages=total_pages,
        page=page,
    )


# ── 회원 가입 승인  GET /admin/members/<id>/approve ──────────────
@admin_bp.route('/members/<int:user_id>/approve')
@admin_required
def admin_member_approve(user_id):
    """선택한 회원의 가입 승인 + 승인 로그 기록."""
    admin_id = session.get('admin_id', 1)
    conn = get_conn()
    cur  = conn.cursor()
    # 상태를 '승인'으로 변경, 거절 사유는 초기화
    cur.execute(
        "UPDATE users SET status = '승인', reject_reason = NULL WHERE user_id = %s",
        (user_id,)
    )
    # 승인 이력 기록
    cur.execute(
        "INSERT INTO user_approval_log (user_id, action, reason, admin_id) VALUES (%s, '승인', NULL, %s)",
        (user_id, admin_id)
    )
    conn.close()
    flash('✅ 회원 가입이 승인되었습니다.', 'success')
    return redirect(url_for('admin.admin_members'))


# ── 회원 가입 거절  POST /admin/members/<id>/reject ───────────────
@admin_bp.route('/members/<int:user_id>/reject', methods=['POST'])
@admin_required
def admin_member_reject(user_id):
    """선택한 회원 가입 거절 + 거절 사유 저장."""
    reason   = request.form.get('reason', '').strip()
    admin_id = session.get('admin_id', 1)

    if not reason:
        flash('거절 사유를 입력해주세요.', 'error')
        return redirect(url_for('admin.admin_members'))

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "UPDATE users SET status = '거절', reject_reason = %s WHERE user_id = %s",
        (reason, user_id)
    )
    cur.execute(
        "INSERT INTO user_approval_log (user_id, action, reason, admin_id) VALUES (%s, '거절', %s, %s)",
        (user_id, reason, admin_id)
    )
    conn.close()
    flash('❌ 회원 가입이 거절되었습니다.', 'error')
    return redirect(url_for('admin.admin_members'))


# =====================================================================
#  공지사항 관리  GET /admin/notice
# =====================================================================
@admin_bp.route('/notice')
@admin_required
def admin_notice():
    """공지사항 목록 — 제목 검색 + 페이지네이션."""
    q      = request.args.get('q', '').strip()
    page   = int(request.args.get('page', 1))
    offset = (page - 1) * PER_PAGE

    conditions, params = [], []
    if q:
        conditions.append("title LIKE %s")
        params.append(f'%{q}%')

    where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

    conn = get_conn()
    cur  = conn.cursor()

    cur.execute(f"SELECT COUNT(*) AS cnt FROM notice {where}", params)
    total_notices = cur.fetchone()['cnt']
    total_pages   = max(1, -(-total_notices // PER_PAGE))

    # is_pinned 컬럼이 notice 테이블에 없으므로 0으로 고정 (추후 구현 예정)
    cur.execute(f"""
        SELECT notice_id, title, created_at, 0 AS is_pinned
        FROM notice
        {where}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """, params + [PER_PAGE, offset])
    notices = cur.fetchall()

    pending_count = get_pending_count(cur)
    conn.close()

    return render_template('admin.html',
        active_tab='notice',
        pending_count=pending_count,
        notices=notices,
        total_notices=total_notices,
        total_pages=total_pages,
        page=page,
    )


# ── 공지사항 등록  POST /admin/notice/create ─────────────────────
@admin_bp.route('/notice/create', methods=['POST'])
@admin_required
def admin_notice_create():
    """새 공지사항 등록."""
    title   = request.form.get('title', '').strip()
    content = request.form.get('content', '').strip()

    if not title or not content:
        flash('제목과 내용을 모두 입력해주세요.', 'error')
        return redirect(url_for('admin.admin_notice'))

    admin_id = session.get('admin_id', 1)    # 작성한 관리자 ID 기록

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO notice (title, content, admin_id) VALUES (%s, %s, %s)",
        (title, content, admin_id)
    )
    conn.close()

    flash('📢 공지사항이 등록되었습니다.', 'success')
    return redirect(url_for('admin.admin_notice'))


# ── 공지사항 삭제  GET /admin/notice/delete/<id> ─────────────────
@admin_bp.route('/notice/delete/<int:notice_id>')
@admin_required
def admin_notice_delete(notice_id):
    """공지사항 삭제."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM notice WHERE notice_id = %s", (notice_id,))
    conn.close()
    flash('공지사항이 삭제되었습니다.', 'info')
    return redirect(url_for('admin.admin_notice'))


# ── 공지사항 수정  GET·POST /admin/notice/edit/<id> ──────────────
@admin_bp.route('/notice/edit/<int:notice_id>', methods=['GET', 'POST'])
@admin_required
def admin_notice_edit(notice_id):
    """GET: 수정 전 데이터를 JSON으로 반환 (JS fetch로 모달에 채움).
    POST: 수정된 내용 저장."""
    conn = get_conn()
    cur  = conn.cursor()

    if request.method == 'POST':
        title   = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        cur.execute(
            "UPDATE notice SET title = %s, content = %s WHERE notice_id = %s",
            (title, content, notice_id)
        )
        conn.close()
        flash('공지사항이 수정되었습니다.', 'success')
        return redirect(url_for('admin.admin_notice'))

    # GET 요청: 수정 모달용 JSON 반환 (admin.js의 openNoticeEditModal에서 fetch)
    cur.execute("SELECT * FROM notice WHERE notice_id = %s", (notice_id,))
    notice = cur.fetchone()
    conn.close()

    if not notice:
        flash('존재하지 않는 공지사항입니다.', 'error')
        return redirect(url_for('admin.admin_notice'))

    from flask import jsonify
    return jsonify({
        'notice_id': notice['notice_id'],
        'title':     notice['title'],
        'content':   notice['content'],
    })


# ── 관리자 로그아웃  GET /admin/logout ──────────────────────────
@admin_bp.route('/logout')
def admin_logout():
    """관리자 세션 종료 후 로그인 페이지로 이동."""
    session.clear()
    flash('로그아웃 되었습니다.', 'info')
    return redirect(url_for('login'))


# =====================================================================
#  사전 확인 관리  GET /admin/precheck
#  비행 전 체크리스트 항목 관리 (추가/삭제)
# =====================================================================
@admin_bp.route('/precheck')
@admin_required
def admin_precheck():
    """사전 확인 항목 목록 표시."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM pre_check ORDER BY sort_order, check_id")
    items = cur.fetchall()
    pending_count = get_pending_count(cur)
    conn.close()
    return render_template('admin.html',
        active_tab='precheck',
        pending_count=pending_count,
        precheck_items=items,
    )


@admin_bp.route('/precheck/create', methods=['POST'])
@admin_required
def admin_precheck_create():
    """새 사전 확인 항목 등록. sort_order는 현재 최댓값+1 자동 설정."""
    icon    = request.form.get('icon', '✅').strip()    # 이모지 아이콘
    content = request.form.get('content', '').strip()   # 체크리스트 내용

    if not content:
        flash('내용을 입력해주세요.', 'error')
        return redirect(url_for('admin.admin_precheck'))

    conn = get_conn()
    cur  = conn.cursor()
    # 현재 최대 정렬 순서 + 1을 새 항목의 순서로 설정
    cur.execute("SELECT COALESCE(MAX(sort_order),0)+1 FROM pre_check")
    next_order = cur.fetchone()['COALESCE(MAX(sort_order),0)+1']
    cur.execute(
        "INSERT INTO pre_check (icon, content, sort_order) VALUES (%s, %s, %s)",
        (icon, content, next_order)
    )
    conn.commit()
    conn.close()
    flash('✅ 사전 확인 항목이 등록되었습니다.', 'success')
    return redirect(url_for('admin.admin_precheck'))


@admin_bp.route('/precheck/delete/<int:check_id>')
@admin_required
def admin_precheck_delete(check_id):
    """사전 확인 항목 삭제."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM pre_check WHERE check_id = %s", (check_id,))
    conn.commit()
    conn.close()
    flash('삭제되었습니다.', 'info')
    return redirect(url_for('admin.admin_precheck'))
