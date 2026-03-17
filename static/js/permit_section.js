/* ================================================================
   permit_section.js  —  비행 허가 신청 패널 전용 스크립트
   역할: 허가 신청 지도 초기화, 좌표 선택, 폼 제출(FormData), 팝업 처리
   주요 함수:
     pmInitMap()          → 비행 허가 탭 활성화 시 소형 지도 초기화 (1회만)
     pmApplyCoord()       → 좌표 유효성 검사 + 지도/폼 동기화
     pmSubmitPermit()     → 신청서 FormData 조립 + /api/permit/submit POST
     pmLoadPrecheck()     → /api/prechecks 에서 사전확인 항목 동적 로드
     pmLoadNotices()      → /api/notices 에서 공지사항 동적 로드
     pmUpdateRadius()     → 슬라이더/입력란 연동 + 원 다시 그리기
     pmDrawCircle()       → OpenLayers로 비행 반경 원 그리기
   주요 변수:
     pmMap           → OpenLayers Map 인스턴스 (최초 1회 생성)
     pmSelectedCoord → 마지막으로 선택된 {coord} 객체
     pmCurrentRadius → 현재 비행 반경 (기본 500m)
     _pmPhotoOn      → 촬영 신청 여부 상태 (boolean)
   의존: ol.js (OpenLayers v8), VWORLD_KEY (main.html 주입)
   주의: pm_ 접두사로 전역 변수/함수명 충돌 방지
================================================================ */

let pmMap           = null;
let pmMarkerSource  = null;
let pmCircleSource  = null;
let pmSelectedCoord = null;
let pmCurrentRadius = 500;

/* ─── 팝업 닫기 ─── */
function pmClosePopup() {
    document.getElementById('pm-popup-overlay').style.display = 'none';
}

/* ─── 팝업 탭 전환 ─── */
document.addEventListener('DOMContentLoaded', () => pmLoadPrecheck());

function pmSwitchTab(tab, btn) {
    document.querySelectorAll('.pm-popup-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.pm-popup-tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('pm-tab-' + tab).classList.add('active');
    if (tab === 'notice') pmLoadNotices();
    if (tab === 'check')  pmLoadPrecheck();
}

/* ─── 사전 확인 DB 동적 로딩 ─── */
let pmPrecheckLoaded = false;

async function pmLoadPrecheck() {
    if (pmPrecheckLoaded) return;
    try {
        const res  = await fetch('/api/prechecks');
        const data = await res.json();
        const list = document.getElementById('pm-check-list');
        if (!data.ok || data.items.length === 0) {
            list.innerHTML = '<li><span class="ico">📋</span><span style="color:var(--text-muted);">등록된 항목이 없습니다.</span></li>';
            return;
        }
        list.innerHTML = data.items.map(item =>
            `<li><span class="ico">${item.icon}</span><span>${item.content}</span></li>`
        ).join('');
        pmPrecheckLoaded = true;
    } catch(e) {
        document.getElementById('pm-check-list').innerHTML =
            '<li><span class="ico">⚠️</span><span style="color:var(--text-muted);">불러오지 못했습니다.</span></li>';
    }
}

/* ─── 팝업 공지사항 DB 동적 로딩 ─── */
let pmNoticesLoaded = false;

async function pmLoadNotices() {
    if (pmNoticesLoaded) return;
    try {
        const res  = await fetch('/api/notices');
        const data = await res.json();
        const list = document.getElementById('pm-notice-list');
        if (!data.ok || data.notices.length === 0) {
            list.innerHTML = '<li style="text-align:center; color:var(--text-muted);">등록된 공지사항이 없습니다.</li>';
            return;
        }
        // 공지사항 각 항목 클릭 시 상세 내용 표시
        list.innerHTML = data.notices.map((n, i) => {
            const safeTitle   = n.title.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
            const safeContent = (n.content||'').replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/\n/g,'\\n');
            return `
            <li style="cursor:pointer; padding:0.55rem 0.5rem; border-bottom:1px solid var(--border);
                       list-style:none; transition:background 0.15s;"
                onmouseover="this.style.background='var(--surface2)'"
                onmouseout="this.style.background='transparent'"
                onclick="pmOpenNoticeDetail('${safeTitle}','${safeContent}','${n.created_at}')">
              <div style="display:flex; align-items:center; gap:0.5rem;">
                ${i === 0 ? '<span class="pm-notice-tag pm-tag-new">NEW</span>' : ''}
                <span style="font-size:0.85rem; color:var(--text); flex:1;">${n.title}</span>
                <span class="pm-notice-date">${n.created_at}</span>
                <span style="color:var(--text-muted); font-size:0.72rem;">▶</span>
              </div>
            </li>`;
        }).join('');
        pmNoticesLoaded = true;
    } catch(e) {
        document.getElementById('pm-notice-list').innerHTML =
            '<li style="text-align:center; color:var(--text-muted);">공지사항을 불러오지 못했습니다.</li>';
    }
}

/* ─── 팝업 내 공지사항 상세 열기/닫기 ─── */
function pmOpenNoticeDetail(title, content, date) {
    document.getElementById('pm-notice-detail-title').textContent   = title;
    document.getElementById('pm-notice-detail-content').textContent = content.replace(/\\n/g, '\n');
    document.getElementById('pm-notice-detail-date').textContent    = '📅 ' + date;
    document.getElementById('pm-notice-detail-overlay').style.display = 'flex';
}
function closePmNoticeDetail() {
    document.getElementById('pm-notice-detail-overlay').style.display = 'none';
}

/* ─── 한국 경위도 범위 상수 ─── */
const KR_LAT_MIN = 33.0, KR_LAT_MAX = 38.7;
const KR_LNG_MIN = 124.5, KR_LNG_MAX = 131.5;

/* ─── 촬영 신청 여부 토글 ─── */
let _pmPhotoOn = false;   // 기본값: OFF (촬영 안함)

function pmTogglePhoto(btn) {
    _pmPhotoOn = !_pmPhotoOn;
    _applyPhotoStyle(btn, _pmPhotoOn);
}

function _applyPhotoStyle(btn, on) {
    if (on) {
        btn.textContent        = '✅ 촬영 신청함';
        btn.style.background   = 'rgba(22,197,94,0.15)';
        btn.style.borderColor  = '#22c55e';
        btn.style.color        = '#22c55e';
    } else {
        btn.textContent        = '🚫 촬영 안함';
        btn.style.background   = 'rgba(100,116,139,0.12)';
        btn.style.borderColor  = '#64748b';
        btn.style.color        = '#94a3b8';
    }
}

function pmGetPhotoRequest() { return _pmPhotoOn; }

/* ─── 지도 클릭 시 뱃지 상태 유지 (기존 호환) ─── */
function pmUpdateZoneBadge() {
    // 지도 클릭 시 토글 버튼 상태를 강제로 바꾸지 않음
}

/* ─── 주소 역지오코딩 ─── */
async function pmFetchAddress(lat, lng) {
    const el = document.getElementById('pm-addr');
    el.value = '조회 중...';
    try {
        const res  = await fetch(`/api/address?lat=${lat}&lng=${lng}`);
        const data = await res.json();
        if (!data.ok) throw new Error(data.msg);
        const parts = data.address.trim().split(/\s+/);
        el.value = parts.slice(0, 3).join(' ');
    } catch {
        el.value = '조회 실패';
    }
}

/* ─── 공통 좌표 검증 & 지도 반영 헬퍼 ─── */
function pmApplyCoord(lat, lng) {
    if (isNaN(lat) || isNaN(lng)) return;

    if (lat < KR_LAT_MIN || lat > KR_LAT_MAX || lng < KR_LNG_MIN || lng > KR_LNG_MAX) {
        clearTimeout(window._pmCoordAlertTimer);
        window._pmCoordAlertTimer = setTimeout(() => {
            const chkLat = parseFloat(document.getElementById('pm-form-lat').value);
            const chkLng = parseFloat(document.getElementById('pm-form-lng').value);
            if (chkLat < KR_LAT_MIN || chkLat > KR_LAT_MAX || chkLng < KR_LNG_MIN || chkLng > KR_LNG_MAX) {
                alert(
                    `⚠️ 한국 범위를 벗어난 좌표입니다.\n\n` +
                    `위도 허용 범위: ${KR_LAT_MIN} ~ ${KR_LAT_MAX}\n` +
                    `경도 허용 범위: ${KR_LNG_MIN} ~ ${KR_LNG_MAX}`
                );
                document.getElementById('pm-form-lat').style.borderColor = '#dc2626';
                document.getElementById('pm-form-lng').style.borderColor = '#dc2626';
            }
        }, 800);
        return;
    }

    clearTimeout(window._pmCoordAlertTimer);
    ['pm-form-lat','pm-form-lng','pm-coord-lat','pm-coord-lng'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.borderColor = '';
    });

    if (!pmMap) return;

    const latStr = lat.toFixed(6);
    const lngStr = lng.toFixed(6);
    const coord  = ol.proj.fromLonLat([lng, lat]);

    pmMap.getView().animate({ center: coord, zoom: 13, duration: 500 });
    pmSelectedCoord = { coord };
    pmMarkerSource.clear();
    pmMarkerSource.addFeature(new ol.Feature(new ol.geom.Point(coord)));
    pmDrawCircle(coord, pmCurrentRadius);

    // 두 입력 영역 모두 동기화
    document.getElementById('pm-form-lat').value  = latStr;
    document.getElementById('pm-form-lng').value  = lngStr;
    document.getElementById('pm-coord-lat').value = latStr;
    document.getElementById('pm-coord-lng').value = lngStr;

    pmFetchAddress(latStr, lngStr);
    pmUpdateZoneBadge();
}

/* ─── 상단 비행지역 입력 → 지도/하단 동기화 ─── */
function pmSyncCoordFromInput() {
    const latVal = document.getElementById('pm-form-lat').value.trim();
    const lngVal = document.getElementById('pm-form-lng').value.trim();
    document.getElementById('pm-coord-lat').value = latVal || '';
    document.getElementById('pm-coord-lng').value = lngVal || '';
    if (!latVal || !lngVal) return;
    pmApplyCoord(parseFloat(latVal), parseFloat(lngVal));
}

/* ─── 하단 선택좌표 직접 입력 → 지도/상단 동기화 ─── */
function pmSyncCoordFromBottom() {
    const latVal = document.getElementById('pm-coord-lat').value.trim();
    const lngVal = document.getElementById('pm-coord-lng').value.trim();
    document.getElementById('pm-form-lat').value = latVal || '';
    document.getElementById('pm-form-lng').value = lngVal || '';
    if (!latVal || !lngVal) return;
    pmApplyCoord(parseFloat(latVal), parseFloat(lngVal));
}

/* ─── 지도 이동 버튼 ─── */
function pmJumpToCoord() {
    const lat = parseFloat(document.getElementById('pm-form-lat').value.trim());
    const lng = parseFloat(document.getElementById('pm-form-lng').value.trim());
    if (isNaN(lat) || isNaN(lng)) {
        alert('올바른 위도/경도를 입력해주세요.\n예) 위도: 37.5665  경도: 126.9780');
        return;
    }
    if (!pmMap) {
        alert('지도가 초기화되지 않았습니다.\n비행 허가 탭을 먼저 클릭해주세요.');
        return;
    }
    pmApplyCoord(lat, lng);
}

/* ─── 사용 목적 토글 ─── */
function pmSelectPurpose(btn) {
    document.querySelectorAll('.pm-purpose-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

/* ─── 기체 타입 토글 ─── */
function pmSelectDrone(btn) {
    document.querySelectorAll('.pm-drone-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

/* ─── 조종자 필드 토글 ─── */
function pmTogglePilot(checkbox) {
    document.getElementById('pm-pilot-fields').style.display = checkbox.checked ? 'none' : 'block';
}

/* ─── 파일 업로드 ─── */
function pmHandleFiles(input) {
    const list = document.getElementById('pm-file-list');
    list.innerHTML = '';
    Array.from(input.files).forEach(f => {
        const div = document.createElement('div');
        div.style.cssText = 'padding:3px 0;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:6px;font-size:0.73rem;';
        div.innerHTML = `<span>📄</span><span>${f.name}</span><span style="color:var(--text-muted);margin-left:auto;">${(f.size/1024).toFixed(1)} KB</span>`;
        list.appendChild(div);
    });
}

/* ─── 반경 슬라이더 ─── */
function pmUpdateRadius(val) {
    pmCurrentRadius = parseInt(val);
    document.getElementById('pm-radius-input').value = val;
    if (pmSelectedCoord && pmCircleSource) {
        pmDrawCircle(pmSelectedCoord.coord, pmCurrentRadius);
    }
}

/* ─── 반경 직접 입력 ─── */
function pmUpdateRadiusDirect(val) {
    let v = parseInt(val);
    if (isNaN(v) || v < 100) v = 100;
    if (v > 5000) v = 5000;
    pmCurrentRadius = v;
    document.getElementById('pm-radius-slider').value = v;
    document.getElementById('pm-radius-input').value  = v;
    if (pmSelectedCoord && pmCircleSource) {
        pmDrawCircle(pmSelectedCoord.coord, pmCurrentRadius);
    }
}

/* ─── 반경 원 그리기 ─── */
function pmDrawCircle(center, radiusMeters) {
    pmCircleSource.clear();
    const res  = ol.proj.getPointResolution('EPSG:3857', 1, center);
    const geom = ol.geom.Polygon.fromCircle(new ol.geom.Circle(center, radiusMeters / res));
    pmCircleSource.addFeature(new ol.Feature(geom));
}

/* ─── 지도 초기화 (showTab('permit')에서 호출, 한 번만 실행) ─── */
function pmInitMap() {
    if (pmMap) { pmMap.updateSize(); return; }

    const baseLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: `http://api.vworld.kr/req/wmts/1.0.0/${VWORLD_KEY}/Base/{z}/{y}/{x}.png`,
            maxZoom: 19,
            crossOrigin: 'anonymous'
        })
    });

    pmMarkerSource = new ol.source.Vector();
    const markerLayer = new ol.layer.Vector({
        source: pmMarkerSource,
        style: new ol.style.Style({
            image: new ol.style.Circle({
                radius: 8,
                fill:   new ol.style.Fill({ color: 'rgba(30,58,95,0.85)' }),
                stroke: new ol.style.Stroke({ color: '#6b9fd4', width: 2 })
            })
        })
    });

    pmCircleSource = new ol.source.Vector();
    const circleLayer = new ol.layer.Vector({
        source: pmCircleSource,
        style: new ol.style.Style({
            fill:   new ol.style.Fill({ color: 'rgba(30,58,95,0.08)' }),
            stroke: new ol.style.Stroke({ color: '#1e3a5f', width: 2, lineDash: [6, 4] })
        })
    });

    pmMap = new ol.Map({
        target: 'permit-vmap',
        layers: [baseLayer, circleLayer, markerLayer],
        view: new ol.View({
            center:  ol.proj.fromLonLat([127.5, 36.5]),
            zoom:    7,
            minZoom: 6,
            maxZoom: 19
        }),
        controls: new ol.Collection([])
    });

    pmMap.on('click', function (evt) {
        const [lng, lat] = ol.proj.toLonLat(evt.coordinate);
        const latStr = lat.toFixed(6);
        const lngStr = lng.toFixed(6);

        pmSelectedCoord = { coord: evt.coordinate };

        document.getElementById('pm-coord-lat').value = latStr;
        document.getElementById('pm-coord-lng').value = lngStr;
        document.getElementById('pm-form-lat').value  = latStr;
        document.getElementById('pm-form-lng').value  = lngStr;

        pmMarkerSource.clear();
        pmMarkerSource.addFeature(new ol.Feature(new ol.geom.Point(evt.coordinate)));
        pmDrawCircle(evt.coordinate, pmCurrentRadius);

        pmFetchAddress(latStr, lngStr);
        pmUpdateZoneBadge();
    });

    setTimeout(() => pmMap.updateSize(), 150);
}

/* ─── 날짜 기본값 설정 ─── */
document.addEventListener('DOMContentLoaded', function () {
    const today = new Date();
    const next  = new Date(today);
    next.setDate(next.getDate() + 1);
    const startEl = document.getElementById('pm-date-start');
    const endEl   = document.getElementById('pm-date-end');
    if (startEl) startEl.value = today.toISOString().split('T')[0];
    if (endEl)   endEl.value   = next.toISOString().split('T')[0];
});

/* ─── 신청서 제출 ─── */
async function pmSubmitPermit() {
    const purposeBtn = document.querySelector('.pm-purpose-btn.active');
    const purpose    = purposeBtn ? purposeBtn.textContent.trim() : '';

    const droneBtn  = document.querySelector('.pm-drone-btn.active');
    const droneType = droneBtn ? droneBtn.textContent.trim() : '';

    const startDate  = document.getElementById('pm-date-start').value;
    const endDate    = document.getElementById('pm-date-end').value;
    const startTime  = (document.getElementById('pm-time-start')  || {}).value || '';
    const endTime    = (document.getElementById('pm-time-end')    || {}).value || '';
    const altitude   = (document.getElementById('pm-altitude')    || {}).value || '';

    const latText    = document.getElementById('pm-coord-lat').value;
    const lngText    = document.getElementById('pm-coord-lng').value;
    const latitude   = (latText === '' || latText === '—') ? null : parseFloat(latText);
    const longitude  = (lngText === '' || lngText === '—') ? null : parseFloat(lngText);

    if (!purpose)                { alert('사용 목적을 선택해주세요.'); return; }
    if (!startDate || !endDate)  { alert('비행 기간을 입력해주세요.'); return; }
    if (!latitude || !longitude) { alert('지도에서 비행 위치를 선택해주세요.'); return; }

    const btn = document.querySelector('.pm-submit-btn');
    btn.disabled = true;
    btn.textContent = '제출 중...';

    try {
        // FormData 방식 — 파일 업로드 포함
        const fd = new FormData();
        fd.append('purpose',          purpose);
        fd.append('drone_type',       droneType);
        fd.append('start_date',       startDate);
        fd.append('end_date',         endDate);
        fd.append('start_time',       startTime);
        fd.append('end_time',         endTime);
        fd.append('flight_altitude',  altitude);
        fd.append('latitude',         latitude);
        fd.append('longitude',        longitude);
        fd.append('radius',           pmCurrentRadius);
        fd.append('photo_request',    pmGetPhotoRequest() ? '1' : '0');

        // 첨부파일 (여러 개)
        const fileInput = document.getElementById('pm-file-input');
        if (fileInput && fileInput.files.length > 0) {
            Array.from(fileInput.files).forEach(f => fd.append('attachments', f));
        }

        const res  = await fetch('/api/permit/submit', { method: 'POST', body: fd });
        const data = await res.json();

        if (data.ok) {
            alert('✅ 비행 허가 신청이 완료되었습니다.\n관리자 검토 후 결과를 안내드립니다.');
            if (pmMarkerSource) pmMarkerSource.clear();
            if (pmCircleSource) pmCircleSource.clear();
            pmSelectedCoord = null;
            ['pm-coord-lat','pm-coord-lng','pm-form-lat','pm-form-lng'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            const addrEl = document.getElementById('pm-addr');
            if (addrEl) addrEl.value = '';
            if (fileInput) fileInput.value = '';
            document.getElementById('pm-file-list').innerHTML = '';
        } else {
            alert('❌ 신청 실패: ' + data.msg);
        }
    } catch (e) {
        alert('❌ 오류가 발생했습니다: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '비행 허가 신청서 제출 →';
    }
}