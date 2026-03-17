/* ================================================================
   weather_section.js  —  날씨 패널 전용 스크립트
   역할: 기상청 API 결과를 받아 비행 가능 여부를 종합 판단하고
         weather_section.html의 요소들을 업데이트함.
   의존: map_section.js (지도 클릭 시 onMapClickWeather 호출)
================================================================ */

/* 기본 위치 좌표: 서울 강남구 (지도 초기 포인트와 동일)
   map_section.js에서도 이 상수를 참조 (typeof DEFAULT_LAT 체크) */
const DEFAULT_LAT = 37.49985;
const DEFAULT_LON = 127.03383;


/* ── 1. 날씨 업데이트 시각 표시 ────────────────────────────────
   서버에서 받은 연/월/일/시를 화면 상단 "업데이트: YYYY.MM.DD HH:00" 형식으로 표시.
   인자가 null이면 현재 브라우저 시각 사용 */
function updateTime(year, month, day, hour) {
    const now = new Date();

    // padStart(4, '0'): 자릿수 부족 시 앞에 0 채우기 (예: 3 → '0003')
    const yyyy = year  != null ? String(year).padStart(4, '0')  : String(now.getFullYear());
    const mm   = month != null ? String(month).padStart(2, '0') : String(now.getMonth() + 1).padStart(2, '0');
    const dd   = day   != null ? String(day).padStart(2, '0')   : String(now.getDate()).padStart(2, '0');
    const hh   = hour  != null ? String(hour).padStart(2, '0')  : '00';

    const el = document.getElementById('weather-time');
    if (el) el.textContent = `업데이트: ${yyyy}.${mm}.${dd} ${hh}:00 기준`;
}


/* ── 2. 비행 가능 여부 종합 판단 함수 ──────────────────────────
   판단 우선순위:
   [1순위] 구역 규칙 (비행금지/제한)
   [2순위] 드론 무게 규칙 (25kg 초과, 7kg 초과 등)
   [3순위] 날씨 기반 판단 (기상청 데이터)
   반환: { status, reason, level } 객체
         level: 'ok'(초록) / 'warn'(노랑) / 'ng'(빨강) */
function evaluateFlightStatus(weatherData, zoneType) {

    // USER_DRONE_WEIGHT: main.html에서 Flask 세션으로 주입된 드론 무게 (없으면 null)
    const weight = (typeof USER_DRONE_WEIGHT !== 'undefined' && USER_DRONE_WEIGHT !== null)
        ? parseFloat(USER_DRONE_WEIGHT)
        : null;

    /* ── [1순위] 구역 기반 판단 ──────────────────────────────── */
    if (zoneType === 'forbidden') {
        // 비행금지구역: 기상과 무관하게 무조건 불가
        return { status: '비행 금지', reason: '비행금지구역(RK P)입니다. 기상과 관계없이 승인 없는 비행은 불가합니다.', level: 'ng' };
    }
    if (zoneType === 'restricted') {
        // 비행제한구역: 관할 기관 승인 필요
        return { status: '비행 승인 필요', reason: '비행제한구역(RK R)입니다. 관할 기관의 승인을 먼저 확인하세요.', level: 'ng' };
    }

    /* ── [2순위] 드론 무게 기반 판단 ─────────────────────────── */
    if (weight !== null) {
        if (weight > 25) {
            // 25kg 초과: 조종자 증명 + 비행 승인 필수 (항공안전법 기준)
            return { status: '비행 승인+자격 필수', reason: `기체 중량 ${weight}kg: 25kg 초과 기체는 비행 승인 및 조종자 증명이 필수입니다.`, level: 'ng' };
        }
        if (weight > 7) {
            // 7kg 초과: 비행 전 신고 및 승인 필요
            return { status: '비행 전 승인 대상', reason: `기체 중량 ${weight}kg: 7kg 초과 기체는 비행 전 신고 및 승인이 필요합니다.`, level: 'warn' };
        }
        if (weight > 2 && zoneType === 'danger') {
            // 위험구역 내 2kg 초과 기체: 주의 권고
            return { status: '비행 위험 주의', reason: `기체 중량 ${weight}kg: 위험구역 내 2kg 초과 기체 비행은 각별히 주의하십시오.`, level: 'warn' };
        }
    }

    /* ── [3순위] 날씨 기반 판단 (서버 app.py flight_status() 결과 활용) ── */
    const ws = weatherData.status;    // '비행 가능' / '비행 주의' / '비행 위험' / '비행 금지' 등

    if (ws === '비행 가능') {
        if (zoneType === 'danger') {
            // 기상은 좋지만 위험구역이면 주의
            return { status: '비행 주의', reason: '기상은 양호하나 현재 위치는 비행위험구역입니다.', level: 'warn' };
        }
        return { status: '비행 가능', reason: '현재 기상 및 구역 조건에서 안전한 비행이 가능합니다.', level: 'ok' };
    } else {
        // 날씨가 좋지 않은 경우: 주의면 warn, 나머지는 ng
        const level = (ws === '비행 주의') ? 'warn' : 'ng';
        return { status: ws, reason: weatherData.reason, level: level };
    }
}


/* ── 3. 날씨 패널 전체 업데이트 ────────────────────────────────
   /api/weather 응답 data와 구역 타입으로 패널을 최신화 */
function updateWeatherPanel(data, zoneType = 'normal') {

    if (!data.ok) {
        // API 실패 시 최소한 상태 제목만 변경
        document.getElementById('status-title').textContent = '날씨 조회 실패';
        return;
    }

    // 비행 가능 여부 종합 판단 (구역 + 무게 + 날씨 우선순위 적용)
    const judgment = evaluateFlightStatus(data, zoneType);

    const iconEl  = document.getElementById('status-icon');
    const titleEl = document.getElementById('status-title');

    /* SVG 아이콘 맵 — 이모지 대신 SVG 사용 이유:
       이모지는 운영체제마다 렌더링이 달라서 정중앙 정렬이 안 됨 */
    const iconMap = {
        // 비행기 아이콘 (ok: 초록)
        ok:   `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="22" height="22"><path d="M21 16v-2l-8-5V3.5A1.5 1.5 0 0 0 11.5 2 1.5 1.5 0 0 0 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z" fill="currentColor"/></svg>`,
        // 경고 삼각형 아이콘 (warn: 노랑)
        warn: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="22" height="22"><path d="M12 2L1 21h22L12 2zm0 3.5L20.5 19h-17L12 5.5zM11 10v4h2v-4h-2zm0 6v2h2v-2h-2z" fill="currentColor"/></svg>`,
        // 금지 원형 아이콘 (ng: 빨강)
        ng:   `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="22" height="22"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`
    };

    // level → CSS 클래스명 매핑
    const classMap = { ok: 'status-ok', warn: 'status-warn', ng: 'status-ng' };

    // 아이콘 교체 + 상태 클래스 변경
    iconEl.innerHTML = iconMap[judgment.level];
    iconEl.className = 'status-icon ' + classMap[judgment.level];

    // 상태 제목 + 이유 텍스트 업데이트
    titleEl.textContent = judgment.status;
    titleEl.className   = 'status-title ' + judgment.level;
    document.getElementById('status-reason').textContent = judgment.reason;

    /* 풍향 각도(deg) → 한글 방위 변환
       8방위: 북(0°), 북동(45°), 동(90°)...
       Math.round(deg / 45) % 8 으로 인덱스 계산 */
    function degToDir(deg) {
        if (deg == null) return '';
        const dirs = ['북', '북동', '동', '남동', '남', '남서', '서', '북서'];
        return dirs[Math.round(deg / 45) % 8];
    }
    const windDir  = (data.wind_dir != null) ? degToDir(data.wind_dir) : '';
    const windText = (data.wind != null)
        ? `${data.wind} m/s${windDir ? ' (' + windDir + ')' : ''}`
        : '-';

    /* 날씨 이모지 결정 (강수 > 바람 > 야간 > 맑음 순서로 판단) */
    function getWeatherIcon(rain, wind, sunrise, sunset) {
        const rain_val = parseFloat(rain) || 0;
        const wind_val = parseFloat(wind) || 0;

        // 현재 시각이 야간(일몰 이후 ~ 일출 이전)인지 판단
        let isNight = false;
        if (sunrise && sunset) {
            const now = new Date();
            const [srH, srM] = sunrise.split(':').map(Number);
            const [ssH, ssM] = sunset.split(':').map(Number);
            const nowMin = now.getHours() * 60 + now.getMinutes();
            const srMin  = srH * 60 + srM;
            const ssMin  = ssH * 60 + ssM;
            isNight = nowMin < srMin || nowMin > ssMin;
        }

        if (rain_val > 5)                    return '⛈';    // 폭우/뇌우
        if (rain_val > 0)                    return '🌧';    // 비
        if (wind_val >= 4)                   return '🌬️';   // 바람 강함
        if (isNight)                         return '🌙';    // 야간
        if (rain_val === 0 && wind_val <= 1) return '☀️';   // 맑음
        return '⛅';    // 구름 조금
    }

    const weatherIcon    = getWeatherIcon(data.rain, data.wind, data.sunrise, data.sunset);
    const iconElWeather  = document.getElementById('weather-icon');
    if (iconElWeather) iconElWeather.textContent = weatherIcon;

    /* ── 메인 헤더 요약 정보 업데이트 ── */
    document.getElementById('weather-wind-main').textContent = windText;
    document.getElementById('weather-rain-main').textContent = (data.rain > 0) ? `강수 ${data.rain}mm` : '강수 없음';
    document.getElementById('weather-stn').textContent       = '📍 ' + data.stn_name;
    updateTime(data.year, data.month, data.day, data.hour);

    /* ── 하단 4개 그리드 (바람/강수량/일출/일몰) 업데이트 ── */
    const windEl    = document.getElementById('info-wind');
    const rainEl    = document.getElementById('info-rain');
    const sunriseEl = document.getElementById('info-sunrise');
    const sunsetEl  = document.getElementById('info-sunset');

    if (windEl)    windEl.textContent    = windText;
    if (rainEl)    rainEl.textContent    = (data.rain != null && data.rain > 0) ? `${data.rain}mm` : '없음';
    if (sunriseEl) sunriseEl.textContent = data.sunrise || '-';
    if (sunsetEl)  sunsetEl.textContent  = data.sunset  || '-';
}


/* ── 4. 날씨 API 호출 ───────────────────────────────────────────
   /api/weather?lat=&lon= 호출 → 서버에서 가장 가까운 관측소 데이터 반환
   zoneType: map_section.js에서 구역 판별 후 전달 (기본값 'normal') */
async function fetchWeather(lat, lon, zoneType = 'normal') {
    try {
        const res  = await fetch(`/api/weather?lat=${lat}&lon=${lon}`);
        const data = await res.json();
        updateWeatherPanel(data, zoneType);
    } catch (e) {
        console.error('날씨 API 오류:', e);
    }
}


/* ── 5. 초기화 ──────────────────────────────────────────────────
   DOM 로드 완료 후 실행:
   1) 현재 시각 표시
   2) 기본 위치(강남) 날씨 로드
   3) window.onMapClickWeather 등록 → map_section.js의 지도 클릭 이벤트와 연동 */
document.addEventListener('DOMContentLoaded', () => {
    updateTime();    // 현재 시각으로 초기 표시

    // 기본 위치(DEFAULT_LAT/LON) 날씨 즉시 조회
    fetchWeather(DEFAULT_LAT, DEFAULT_LON);

    // 지도 클릭 시 map_section.js가 이 함수를 호출해서 날씨를 업데이트
    // (두 파일이 window 객체를 통해 통신하는 패턴)
    window.onMapClickWeather = fetchWeather;
});
