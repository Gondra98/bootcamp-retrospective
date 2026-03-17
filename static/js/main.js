/* ================================================================
   main.js  —  메인 페이지 전용 스크립트
   역할: 탭 전환(지도/허가/공지), 공지사항 동적 로드 및 모달 표시.
   의존: IS_LOGGED_IN, IS_ADMIN (main.html에서 Flask 세션 값으로 주입)
================================================================ */

/* ── 탭 전환 함수 ───────────────────────────────────────────────
   tab: 'map' / 'permit' / 'notice'
   btn: 클릭된 nav-tab 버튼 요소 (active 클래스 이동에 사용) */
function showTab(tab, btn) {

    // 비행 허가 탭: 비로그인 상태면 로그인 페이지로 강제 이동
    // IS_LOGGED_IN, IS_ADMIN은 main.html의 <script>에서 Jinja2로 주입됨
    if (tab === 'permit' && !IS_LOGGED_IN && !IS_ADMIN) {
        alert('비행 허가 신청은 로그인 후 이용 가능합니다.');
        location.href = '/login';
        return;
    }

    // 모든 탭 버튼에서 active 클래스 제거 후 클릭된 버튼에만 추가
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');

    // 패널 요소 참조 (id로 접근)
    const panelMap     = document.getElementById('panel-map');
    const panelWeather = document.getElementById('panel-weather');
    const panelPermit  = document.getElementById('panel-permit');
    const panelNotice  = document.getElementById('panel-notice');

    // 모든 패널 숨기기 (공지사항은 display 대신 active 클래스로 제어)
    panelMap.style.display     = 'none';
    panelWeather.style.display = 'none';
    panelPermit.style.display  = 'none';
    panelNotice.classList.remove('active');

    if (tab === 'map') {
        // 지도 + 날씨 패널을 함께 표시 (flex 레이아웃으로 나란히 배치)
        panelMap.style.display     = 'flex';
        panelWeather.style.display = 'flex';

        /* OpenLayers 주의사항:
           지도가 숨겨진 상태(display:none)에서는 컨테이너 크기를 0으로 인식함.
           탭이 다시 보인 후 50ms 뒤에 updateSize()를 호출해 지도 크기를 재계산 */
        setTimeout(() => {
            if (typeof vMap !== 'undefined' && vMap) vMap.updateSize();
        }, 50);

    } else if (tab === 'permit') {
        panelPermit.style.display = 'block';
        pmInitMap();    // permit_section.js의 지도 초기화 (최초 1회만 실행)

    } else if (tab === 'notice') {
        panelNotice.classList.add('active');    // CSS display:flex 활성화
        loadNotices();    // 공지사항 API 호출 (이미 로드됐으면 재호출 안 함)
    }
}


/* ── 공지사항 동적 로딩 ─────────────────────────────────────────
   /api/notices 를 fetch로 호출해 공지사항 목록을 HTML로 렌더링.
   noticesLoaded 플래그: 탭을 여러 번 클릭해도 한 번만 API 호출 */
let noticesLoaded = false;

async function loadNotices() {
    if (noticesLoaded) return;    // 이미 로드됐으면 스킵

    try {
        const res  = await fetch('/api/notices');
        const data = await res.json();
        const list = document.getElementById('notice-list');

        if (!data.ok || data.notices.length === 0) {
            list.innerHTML = '<div class="notice-item" style="justify-content:center; color:var(--text-muted); font-size:0.83rem;">등록된 공지사항이 없습니다.</div>';
            return;
        }

        // 공지사항 목록 HTML 생성 (배열 map으로 각 항목을 템플릿 리터럴로 변환)
        list.innerHTML = data.notices.map((n, i) => `
            <div class="notice-item" onclick="openNoticeModal('${escapeHtml(n.title)}', '${escapeHtml(n.content)}', '${n.created_at}')">
                ${i === 0 ? '<span class="notice-badge badge-new">NEW</span>' : ''}
                <span class="notice-title">${n.title}</span>
                <span class="notice-date">${n.created_at}</span>
            </div>
        `).join('');

        noticesLoaded = true;    // 다음 탭 전환에서 재호출 방지

    } catch (e) {
        document.getElementById('notice-list').innerHTML =
            '<div class="notice-item" style="justify-content:center; color:var(--text-muted); font-size:0.83rem;">공지사항을 불러오지 못했습니다.</div>';
    }
}


/* ── HTML 이스케이프 유틸리티 ────────────────────────────────────
   onclick 속성 내 문자열에 특수문자가 있으면 JS 오류가 발생함.
   작은따옴표(')와 줄바꿈(\n)을 이스케이프 처리 */
function escapeHtml(str) {
    return str.replace(/'/g, "\\'").replace(/\n/g, '\\n');
}


/* ── 공지사항 상세 모달 열기 ────────────────────────────────────
   클릭된 공지 항목의 제목, 내용, 날짜를 모달에 채우고 표시 */
function openNoticeModal(title, content, date) {
    document.getElementById('modal-title').textContent   = title;
    // '\\n'을 실제 줄바꿈(\n)으로 변환 (escapeHtml에서 치환한 것을 복원)
    document.getElementById('modal-content').textContent = content.replace(/\\n/g, '\n');
    document.getElementById('modal-date').textContent    = '📅 ' + date;

    // 모달 오버레이 표시 (display:flex로 중앙 정렬)
    const overlay = document.getElementById('notice-modal');
    overlay.style.display = 'flex';
}

/* ── 공지사항 모달 닫기 ─────────────────────────────────────────
   모달 오버레이를 숨김 처리 */
function closeNoticeModal() {
    document.getElementById('notice-modal').style.display = 'none';
}

/* ── ESC 키로 모달 닫기 ─────────────────────────────────────────
   사용자 편의를 위해 키보드 ESC로도 닫을 수 있게 처리 */
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeNoticeModal();
});
