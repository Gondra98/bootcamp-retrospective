/* ================================================================
   admin.js  —  관리자 페이지 전용 스크립트
   역할: 상세 모달, 거절 모달, 일괄 처리, 공지사항 수정, 토스트 알림
================================================================ */

/* ── 날짜 input 연도 4자리 초과 입력 방지 ─────────────────────
   브라우저 date input에서 연도를 5자리 이상 입력할 수 있는 버그 방지.
   oninput="limitYear(this)" 으로 연결 */
function limitYear(input) {
    if (!input.value) return;
    const parts = input.value.split('-');
    if (parts[0] && parts[0].length > 4) {
        parts[0]    = parts[0].slice(0, 4);    // 연도를 앞 4자리만 유지
        input.value = parts.join('-');
    }
}

/* ── 관리자 지도 날짜 유효성 검사 ────────────────────────────────
   시작 날짜가 종료 날짜보다 늦으면 토스트 경고 후 제출 차단.
   admin.html의 날짜 필터 form에서 onsubmit="return validateAdminMapDate(this)" 로 연결 */
function validateAdminMapDate(form) {
    const from = form.querySelector('[name="date_from"]').value;
    const to   = form.querySelector('[name="date_to"]').value;
    if (from && to && from > to) {
        showToast('시작 날짜는 종료 날짜보다 이전이어야 합니다.', 'warn');
        return false;    // false 반환 → 폼 submit 취소
    }
    return true;
}

/* ── 토스트 알림 ────────────────────────────────────────────────
   화면 우측 상단에 3초간 메시지를 띄우는 알림.
   id="toastWrap" 컨테이너에 동적으로 div를 추가하는 방식.
   type: 'info' / 'success' / 'error' / 'warn' → CSS 클래스로 색상 구분 */
function showToast(msg, type) {
    type = type || 'info';
    const wrap = document.getElementById('toastWrap');
    if (!wrap) return;

    const toast = document.createElement('div');
    toast.className   = 'toast ' + type;    // admin.css에서 색상 스타일 적용
    toast.textContent = msg;
    wrap.appendChild(toast);

    // 3초 후 자동으로 DOM에서 제거
    setTimeout(function() { toast.remove(); }, 3000);
}

/* ── 상세 모달 열기 (data-* 속성 래퍼) ──────────────────────────
   HTML 버튼의 data-* 속성에서 값을 읽어 openDetailModal()에 전달.
   admin.html 목록 행의 "상세" 버튼에 onclick="openDetailModalFromBtn(this)" 연결 */
function openDetailModalFromBtn(btn) {
    openDetailModal(
        parseInt(btn.dataset.pid)       || 0,     // 허가 신청 ID
        btn.dataset.name                || '',     // 신청자 이름
        btn.dataset.lid                 || '',     // 신청자 로그인 ID
        btn.dataset.phone               || '-',    // 연락처
        btn.dataset.area                || '',     // 비행 목적
        btn.dataset.period              || '',     // 비행 기간
        btn.dataset.stime               || '-',    // 시작 시간
        btn.dataset.etime               || '-',    // 종료 시간
        btn.dataset.drone               || '-',    // 드론 종류
        parseInt(btn.dataset.altitude)  || 0,      // 비행 고도
        parseInt(btn.dataset.radius)    || 500,    // 비행 반경
        btn.dataset.coord               || '',     // 좌표
        btn.dataset.reason              || '',     // 거절 사유
        parseInt(btn.dataset.photo)     || 0       // 촬영 신청 여부
    );
}

/* ── 상세 모달 내용 생성 및 열기 ────────────────────────────────
   허가 신청 상세 정보를 모달 HTML로 구성하고 표시.
   row() 함수로 라벨+값 행을 반복 생성하는 패턴 */
function openDetailModal(permitId, name, loginId, phone, area, period,
                         startTime, endTime, drone, altitude, radius, coord, rejectReason, photoRequest) {
    var html = '';

    /* row(label, value, accent): 라벨-값 쌍 div를 생성하는 내부 함수
       accent=true면 값에 accent 클래스(파란색 강조) 추가 */
    function row(label, value, accent) {
        return '<div class="detail-item">' +
               '<div class="detail-label">' + label + '</div>' +
               '<div class="detail-value' + (accent ? ' accent' : '') + '">' + (value || '—') + '</div>' +
               '</div>';
    }

    // 기본 정보 행들 생성
    html += row('신청자',   name);
    html += row('아이디',   loginId,  true);    // 강조 표시
    html += row('연락처',   phone);
    html += row('비행 목적', area);
    html += row('비행 기간', period);
    html += row('비행 시간', startTime + ' ~ ' + endTime);
    html += row('드론 종류', drone);
    html += row('비행 고도', altitude ? altitude + ' m' : '—');
    html += row('비행 반경', radius + ' m');
    html += row('좌표',     coord,    true);
    html += row('촬영 신청', photoRequest ? '✅ 촬영 신청함' : '🚫 촬영 안함');

    // 거절 사유 (전체 너비, 사유 있으면 빨간색, 없으면 회색)
    html += '<div class="detail-item" style="grid-column:1/-1;">' +
                '<div class="detail-label">거절 사유</div>' +
                '<div class="detail-value" style="color:' +
                    (rejectReason ? 'var(--red)' : 'var(--text-muted)') + ';">' +
                    (rejectReason || '—') +
                '</div>' +
            '</div>';

    // 첨부파일 영역 (비동기로 채움, 초기엔 로딩 중 표시)
    html += '<div class="detail-item" style="grid-column:1/-1;">' +
                '<div class="detail-label">첨부파일</div>' +
                '<div class="detail-value" id="detail-files-wrap">' +
                    '<span style="color:var(--text-muted);font-size:0.82rem;">불러오는 중...</span>' +
                '</div>' +
            '</div>';

    document.getElementById('detailContent').innerHTML = html;
    document.getElementById('detailModal').classList.add('open');
    loadAdminPermitFiles(permitId);    // 첨부파일 목록 비동기 로드
}

/* ── 관리자 첨부파일 목록 조회 ──────────────────────────────────
   /api/my_permits/<id>/files 를 fetch로 호출해 파일 목록을 렌더링.
   관리자 세션이면 본인 여부 무관하게 조회 가능 */
async function loadAdminPermitFiles(permitId) {
    const wrap = document.getElementById('detail-files-wrap');
    if (!wrap) return;
    try {
        const res  = await fetch('/api/my_permits/' + permitId + '/files');
        const data = await res.json();

        if (!data.ok || data.files.length === 0) {
            wrap.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem;">첨부파일 없음</span>';
            return;
        }

        // 파일마다 다운로드 링크 포함 행 생성
        var rows = '';
        data.files.forEach(function(f) {
            var kb = f.file_size ? ((f.file_size / 1024).toFixed(1) + ' KB') : '';
            rows += '<div style="display:flex;align-items:center;gap:0.5rem;padding:0.28rem 0;' +
                        'border-bottom:1px solid var(--border);font-size:0.82rem;">' +
                        '<span>📄</span>' +
                        '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' +
                            f.original_name +
                        '</span>' +
                        '<span style="color:var(--text-muted);font-size:0.72rem;">' + kb + '</span>' +
                        // download 속성: 클릭 시 저장 다이얼로그 열림
                        '<a href="' + f.download_url + '" download="' + f.original_name + '"' +
                           ' class="btn btn-ghost btn-sm"' +
                           ' style="font-size:0.72rem;padding:0.15rem 0.5rem;white-space:nowrap;">' +
                            '⬇ 다운로드' +
                        '</a>' +
                    '</div>';
        });
        wrap.innerHTML = rows;
    } catch(e) {
        wrap.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem;">파일 목록 조회 실패</span>';
    }
}

/* ── 상세 모달 닫기 ─────────────────────────────────────────── */
function closeDetailModal() {
    document.getElementById('detailModal').classList.remove('open');
}

/* ── 회원 거절 모달 열기/닫기 ──────────────────────────────────
   rejectForm의 action URL을 동적으로 설정해서 재사용 */
function openRejectModal(userId, userName) {
    document.getElementById('rejectTargetName').textContent = userName;
    document.getElementById('rejectReason').value           = '';    // 사유 초기화
    document.getElementById('rejectForm').action            = '/admin/members/' + userId + '/reject';
    document.getElementById('rejectModal').classList.add('open');
}
function closeRejectModal() {
    document.getElementById('rejectModal').classList.remove('open');
}

/* ── 허가 신청 거절 모달 열기/닫기 ─────────────────────────────
   회원 거절 모달과 별도 (다른 form, 다른 URL) */
function openPermitRejectModal(permitId, name) {
    document.getElementById('permitRejectTargetName').textContent = name;
    document.getElementById('permitRejectReason').value           = '';
    document.getElementById('permitRejectForm').action            = '/admin/approval/' + permitId + '/reject';
    document.getElementById('permitRejectModal').classList.add('open');
}
function closePermitRejectModal() {
    document.getElementById('permitRejectModal').classList.remove('open');
}

/* ── 거절 사유 확인 모달 (읽기 전용) ───────────────────────────
   이미 거절된 항목의 사유를 팝업으로 표시 */
function openReasonModal(userName, reason) {
    document.getElementById('reasonTargetName').textContent = userName;
    document.getElementById('reasonContent').textContent    = reason;
    document.getElementById('reasonModal').classList.add('open');
}
function closeReasonModal() {
    document.getElementById('reasonModal').classList.remove('open');
}

/* ── 일괄 처리 체크박스 ─────────────────────────────────────────
   전체 선택/해제 + 하단 일괄 처리 바 표시/숨김 */

// 전체 선택 체크박스 클릭 → 모든 row-cb 체크박스 상태 동기화
function toggleSelectAll(cb) {
    document.querySelectorAll('.row-cb').forEach(function(c) { c.checked = cb.checked; });
    updateBulkBar();
}

// 선택된 개수에 따라 하단 일괄 처리 바 표시/숨김
function updateBulkBar() {
    const n   = document.querySelectorAll('.row-cb:checked').length;
    const bar = document.getElementById('bulkBar');
    if (!bar) return;
    document.getElementById('bulkCount').textContent = n + '개';
    bar.classList.toggle('show', n > 0);    // 1개 이상 선택 시 show 클래스 추가
}

// 일괄 승인/거절 처리: bulkAction hidden input에 값 넣고 form submit
function submitBulk(action) {
    if (!document.querySelectorAll('.row-cb:checked').length) return;
    document.getElementById('bulkAction').value = action;
    document.getElementById('bulkForm').submit();
}

// 일괄 삭제: confirm 팝업으로 한 번 더 확인 후 처리
function submitBulkDelete() {
    const n = document.querySelectorAll('.row-cb:checked').length;
    if (!n) return;
    if (!confirm('선택한 ' + n + '건을 삭제하시겠습니까?\n삭제된 데이터는 복구할 수 없습니다.')) return;
    document.getElementById('bulkAction').value = 'delete';
    document.getElementById('bulkForm').submit();
}

/* ── 공지사항 수정 모달 열기/닫기 ──────────────────────────────
   /admin/notice/edit/<id> GET 요청으로 공지사항 데이터를 JSON으로 받아
   수정 모달 폼에 채워 넣는 방식 */
async function openNoticeEditModal(noticeId) {
    try {
        const res  = await fetch('/admin/notice/edit/' + noticeId);
        const data = await res.json();
        document.getElementById('noticeEditTitle').value   = data.title;
        document.getElementById('noticeEditContent').value = data.content;
        // form action URL을 수정 대상 ID로 동적 설정
        document.getElementById('noticeEditForm').action   = '/admin/notice/edit/' + noticeId;
        document.getElementById('noticeEditModal').classList.add('open');
    } catch(e) {
        showToast('공지사항 정보를 불러오지 못했습니다.', 'error');
    }
}
function closeNoticeEditModal() {
    document.getElementById('noticeEditModal').classList.remove('open');
}

/* ── Fallback 함수 정의 ──────────────────────────────────────────
   다른 스크립트에서 참조할 수 있는 함수들이 없을 때
   undefined 에러 방지를 위해 빈 함수로 대체 */
if (typeof window.togglePins      === 'undefined') window.togglePins      = function() {};
if (typeof window.adminToggleZone === 'undefined') window.adminToggleZone = function() {};

/* ── Flask flash 메시지 → 토스트로 변환 ────────────────────────
   서버에서 flash()로 등록된 메시지가 hidden div로 렌더링되어 있으면
   JS가 이를 감지해 showToast()로 표시 (HTML 요소 없이 깔끔한 알림) */
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.flash-msg').forEach(function(el) {
        // CSS 클래스로 메시지 타입 판단
        const type = el.classList.contains('flash-success') ? 'success'
                   : el.classList.contains('flash-error')   ? 'error' : 'info';
        showToast(el.textContent.trim(), type);
    });
});
