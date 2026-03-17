/* ================================================================
   auth_forms.js  —  로그인 / 회원가입 / 마이페이지 공통 폼 유틸리티
   역할: 폼 입력 자동 포맷팅(생년월일, 전화번호, cm 단위)과
         회원가입 필수 항목 유효성 검사를 담당.
   적용: login.html, register.html, mypage.html
================================================================ */

/* ── 달력 버튼 클릭 → 숨겨진 date picker 열기 ──────────────────
   register.html / mypage.html에서 📅 버튼 클릭 시 호출.
   input[type="date"]는 기본 UI가 작아서 숨김 처리 후 버튼으로 여는 방식 사용. */
function openBirthPicker() {
    const picker = document.getElementById('birth-picker');
    if (!picker) return;
    try {
        picker.showPicker();    // 브라우저 달력 팝업 강제 오픈 (최신 브라우저만 지원)
    } catch(e) {}               // 구형 브라우저 오류 무시
}

/* ── 달력 선택 → 텍스트 입력란에 반영 ──────────────────────────
   birth-picker(숨김 date input)에서 날짜를 선택하면
   birth-input(텍스트 input)에 값을 복사 */
document.addEventListener('DOMContentLoaded', function () {
    const picker = document.getElementById('birth-picker');
    if (picker) {
        picker.addEventListener('change', function () {
            const input = document.getElementById('birth-input');
            if (input) input.value = this.value;    // YYYY-MM-DD 형식으로 자동 채움
        });
    }
});

/* ── 생년월일 자동 하이픈 포맷팅 ────────────────────────────────
   사용자가 숫자만 입력하면 자동으로 YYYY-MM-DD 형식으로 변환.
   예: 19991231 → 1999-12-31
   oninput="formatBirth(this)"으로 연결 */
function formatBirth(input) {
    // 숫자 이외의 문자 모두 제거
    let val = input.value.replace(/[^0-9]/g, '');

    if (val.length <= 4) {
        // 연도만 입력 중: 그대로 표시
        input.value = val;
    } else if (val.length <= 6) {
        // 연도+월 입력 중: YYYY-MM 형식
        input.value = val.slice(0, 4) + '-' + val.slice(4);
    } else {
        // 연도+월+일 입력 완료: YYYY-MM-DD 형식 (최대 8자리)
        input.value = val.slice(0, 4) + '-' + val.slice(4, 6) + '-' + val.slice(6, 8);
    }
}

/* ── 크기 입력 시 'cm' 단위 자동 추가 ──────────────────────────
   숫자를 입력하면 뒤에 자동으로 'cm'를 붙여줌.
   커서 위치도 숫자 뒤로 유지 (cm 뒤에 오지 않도록) */
function formatCm(input) {
    let val = input.value.replace(/[^0-9.]/g, '');    // 숫자와 소수점만 남김
    input.value = val ? val + 'cm' : '';
    const pos = val.length;                            // 커서를 'cm' 앞(숫자 뒤)에 위치
    input.setSelectionRange(pos, pos);
}

/* ── 전화번호 자동 하이픈 포맷팅 ────────────────────────────────
   숫자만 입력하면 자동으로 010-XXXX-XXXX 형식으로 변환.
   예: 01012341234 → 010-1234-1234
   oninput="pmFormatPhone(this)"으로 연결 */
function pmFormatPhone(input) {
    let val = input.value.replace(/[^0-9]/g, '');    // 숫자만 남김

    if (val.length <= 3) {
        // 010 까지만 입력: 그대로 표시
        input.value = val;
    } else if (val.length <= 7) {
        // 010-XXXX 까지 입력: 첫 하이픈만 추가
        input.value = val.slice(0, 3) + '-' + val.slice(3);
    } else {
        // 010-XXXX-XXXX 완성: 하이픈 2개 추가 (최대 11자리)
        input.value = val.slice(0, 3) + '-' + val.slice(3, 7) + '-' + val.slice(7, 11);
    }
}

/* ── 토스트 알림 표시 (회원가입 전용) ───────────────────────────
   화면 상단 가운데에 빨간색 팝업 메시지를 2.5초간 표시 후 자동 사라짐.
   id="reg-toast"로 중복 생성 방지 (기존 것 먼저 제거) */
function showRegToast(msg) {
    const prev = document.getElementById('reg-toast');
    if (prev) prev.remove();    // 이미 표시 중인 토스트 제거

    const toast = document.createElement('div');
    toast.id          = 'reg-toast';
    toast.textContent = msg;

    // 인라인 스타일로 토스트 디자인 (CSS 파일 의존 없이 독립 동작)
    Object.assign(toast.style, {
        position:     'fixed',
        top:          '28px',
        left:         '50%',
        transform:    'translateX(-50%)',     // 가로 중앙 정렬
        background:   'var(--red, #dc2626)', // CSS 변수 없으면 기본값 사용
        color:        '#fff',
        fontFamily:   "'Noto Sans KR', sans-serif",
        fontSize:     '0.82rem',
        fontWeight:   '600',
        padding:      '10px 20px',
        borderRadius: '8px',
        boxShadow:    '0 4px 20px rgba(220,38,38,0.35)',
        zIndex:       '9999',
        whiteSpace:   'nowrap',
        animation:    'toastIn 0.25s ease',   // common.css에 keyframes 정의됨
    });
    document.body.appendChild(toast);

    // 2.5초 후 페이드아웃 후 DOM에서 제거
    setTimeout(() => {
        toast.style.transition = 'opacity 0.3s';
        toast.style.opacity    = '0';
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}

/* ── 입력 필드 에러 표시 / 해제 ────────────────────────────────
   name 속성으로 필드를 찾아 빨간 테두리와 그림자를 추가/제거.
   hasError=true: 에러 표시 / false: 에러 해제 */
function setFieldError(name, hasError) {
    const input = document.querySelector(`[name="${name}"]`);
    if (!input) return;
    input.style.borderColor = hasError ? '#dc2626' : '';
    input.style.boxShadow   = hasError ? '0 0 0 3px rgba(220,38,38,0.1)' : '';
}

/* ── 회원가입 폼 유효성 검사 ─────────────────────────────────────
   필수 항목 중 하나라도 비어있으면 submit을 막고 첫 번째 빈 항목에 포커스 */
document.addEventListener('DOMContentLoaded', function () {
    const registerForm = document.getElementById('register-form');
    if (!registerForm) return;    // register.html이 아니면 실행 안 함

    // 필수 항목 목록 (name 속성과 한글 레이블)
    const requiredFields = [
        { name: 'name',     label: '이름' },
        { name: 'birth',    label: '생년월일' },
        { name: 'phone',    label: '전화번호' },
        { name: 'login_id', label: '아이디' },
        { name: 'password', label: '비밀번호' },
    ];

    // 각 필드에 입력 이벤트 → 에러 테두리 자동 해제 (타이핑 시작하면 바로 빨간 테두리 제거)
    requiredFields.forEach(f => {
        const input = document.querySelector(`[name="${f.name}"]`);
        if (input) input.addEventListener('input', () => setFieldError(f.name, false));
    });

    // 폼 제출 시 유효성 검사
    registerForm.addEventListener('submit', function (e) {
        let firstEmpty = null;    // 첫 번째로 비어있는 필드 저장

        // 모든 에러 표시 초기화
        requiredFields.forEach(f => setFieldError(f.name, false));

        // 필수 항목 순서대로 체크
        for (const field of requiredFields) {
            const input = document.querySelector(`[name="${field.name}"]`);
            if (!input || !input.value.trim()) {
                setFieldError(field.name, true);    // 빨간 테두리 추가
                if (!firstEmpty) firstEmpty = { input, label: field.label };    // 첫 빈 항목 기억
            }
        }

        // 빈 항목이 있으면 제출 취소 + 안내 메시지 + 해당 입력창 포커스
        if (firstEmpty) {
            e.preventDefault();
            showRegToast(`⚠️ ${firstEmpty.label}을(를) 입력해 주세요.`);
            firstEmpty.input.focus();
        }
    });
});
