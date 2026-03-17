/* ================================================================
   map_section.js  —  OpenLayers 지도 초기화 및 구역 레이어 관리
   역할: Vworld WMTS 배경지도 + 비행금지구역 폴리곤 + 허가 핀 마커
   주요 함수:
     initMap()            → OpenLayers 지도 생성 (DOMContentLoaded에서 호출)
     loadZoneData()       → /api/zones 에서 구역 폴리곤 로드
     loadUserPermits()    → /api/my_permits 에서 내 허가 신청 구역 로드 (일반 유저)
     loadAllPermits()     → /api/all_permits 에서 전체 허가 구역 로드 (관리자)
     onMapClick()         → 클릭 좌표 → 구역 판별 + 날씨 API 트리거
     showInfoPanel()      → 좌측 상단 위치 정보 패널 업데이트
     toggleLayer()        → 구역 유형별 레이어 표시/숨김
     mapZoomIn/Out/Reset()→ 줌 컨트롤 버튼 핸들러
   의존: ol.js (OpenLayers v8), VWORLD_KEY (main.html 주입),
         IS_LOGGED_IN, IS_ADMIN (main.html 주입)
================================================================ */

// ────────────────────────────────────
// 1. 전역 상태
// ────────────────────────────────────
const KR_CENTER = ol.proj.fromLonLat([127.5, 36.5]);
const INIT_ZOOM = 7;
let pinSource = null;
let pinLayer  = null;

// ── 내 허가 신청 구역 레이어 (일반 유저) ────────────────
let userPermitSource  = null;
let userPermitLayer   = null;
let hasUserPermits    = false;   // 이력 유무 플래그

// ── 관리자 허가 현황 레이어 ──────────────────────────────
const adminPermitSources = { approved: null, pending: null, rejected: null };
const adminPermitLayers  = { approved: null, pending: null, rejected: null };

// ── 허가 신청 핀 마커 레이어 (유저/관리자 공통) ──────────
let permitPinSource = null;
let permitPinLayer  = null;

// 구역 유형별 스타일
const ZONE_STYLES = {
    forbidden:  makeStyle('rgba(239,68,68,0.3)',  '#ef4444'),
    restricted: makeStyle('rgba(245,158,11,0.3)', '#f59e0b'),
    danger:     makeStyle('rgba(168,85,247,0.3)', '#a855f7'),
    permit:     makeStyle('rgba(34,197,94,0.3)',  '#22c55e')
};

// 구역 유형 → 한글 라벨
const ZONE_LABEL = {
    forbidden:  '🚫 비행금지구역',
    restricted: '⚠️ 비행제한구역',
    danger:     '☢️ 비행위험구역',
    permit:     '✅ 허가요청구역'
};

// 레이어 가시성 상태
const layerVisible = {
    forbidden:  true,
    restricted: true,
    danger:     true,
    permit:     true
};

const zoneSources = {};  // 구역별 벡터 소스
const zoneLayers  = {};  // 구역별 벡터 레이어

let vMap = null;  // 지도 객체 (initMap 이후 할당)


// ────────────────────────────────────
// 2. 스타일 생성 함수
// ────────────────────────────────────
function makeStyle(fillColor, strokeColor) {
    return new ol.style.Style({
        fill:   new ol.style.Fill({ color: fillColor }),
        stroke: new ol.style.Stroke({ color: strokeColor, width: 2 })
    });
}


// ────────────────────────────────────
// 3. 지도 초기화
// ────────────────────────────────────
function initMap() {
    // vworld WMTS 배경 타일
    const baseLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: `https://api.vworld.kr/req/wmts/1.0.0/${VWORLD_KEY}/Base/{z}/{y}/{x}.png`,
            maxZoom: 19,
            crossOrigin: 'anonymous'
        })
    });

    // 구역 유형별 벡터 레이어 생성
    ['forbidden', 'restricted', 'danger', 'permit'].forEach(type => {
        zoneSources[type] = new ol.source.Vector();
        zoneLayers[type]  = new ol.layer.Vector({
            source: zoneSources[type],
            style:  ZONE_STYLES[type]
        });
    });

    // 지도 생성
    vMap = new ol.Map({
        target: 'vmap',
        layers: [baseLayer, ...Object.values(zoneLayers)],
        view: new ol.View({
            center:  KR_CENTER,
            zoom:    INIT_ZOOM,
            minZoom: 6,
            maxZoom: 19
        }),
        controls: new ol.Collection([])
    });

    // ── 핀 마커 레이어 ──────────────────────────
    pinSource = new ol.source.Vector();
    pinLayer  = new ol.layer.Vector({
        source: pinSource,
        style: new ol.style.Style({
            image: new ol.style.Icon({
                anchor:    [0.5, 1.0],
                anchorXUnits: 'fraction',
                anchorYUnits: 'fraction',
                src:   'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/images/marker-icon.png',
                scale: 1.0,
            })
        }),
        zIndex: 999
    });
    vMap.addLayer(pinLayer);

    // ── 내 허가 신청 구역 레이어 (일반 유저) ────
    userPermitSource = new ol.source.Vector();
    userPermitLayer  = new ol.layer.Vector({
        source: userPermitSource,
        zIndex: 10
    });
    vMap.addLayer(userPermitLayer);

    // ── 관리자 허가 현황 레이어 (승인/대기/거절) ─
    const adminColorMap = {
        approved: { fill: 'rgba(34,197,94,0.15)',  stroke: '#22c55e' },
        pending:  { fill: 'rgba(56,189,248,0.12)', stroke: '#38bdf8' },
        rejected: { fill: 'rgba(239,68,68,0.10)',  stroke: '#ef4444' }
    };
    ['approved', 'pending', 'rejected'].forEach(status => {
        const c = adminColorMap[status];
        adminPermitSources[status] = new ol.source.Vector();
        adminPermitLayers[status]  = new ol.layer.Vector({
            source: adminPermitSources[status],
            style: new ol.style.Style({
                fill:   new ol.style.Fill({ color: c.fill }),
                stroke: new ol.style.Stroke({ color: c.stroke, width: 2, lineDash: [6, 4] })
            }),
            zIndex: 9
        });
        vMap.addLayer(adminPermitLayers[status]);
    });

    // ── 허가 신청 핀 마커 레이어 ────────────────────────
    permitPinSource = new ol.source.Vector();
    permitPinLayer  = new ol.layer.Vector({
        source: permitPinSource,
        zIndex: 20
    });
    vMap.addLayer(permitPinLayer);

    // 이벤트 바인딩
    vMap.on('pointermove', onPointerMove);
    vMap.on('click', (e) => {
        // 1) 허가 핀 클릭 체크 → 지도 아래 카드 표시
        let hitPermit = false;
        vMap.forEachFeatureAtPixel(e.pixel, (feat) => {
            if (feat.get('_permit') && !hitPermit) {
                hitPermit = true;
                showPinCard(feat.get('_permit'), e.pixel);
            }
        }, { layerFilter: l => l === permitPinLayer });

        if (!hitPermit) {
            closePinCard();
            onMapClick(e);   // 기존 클릭 로직
        }
    });
}


// ────────────────────────────────────
// 4. 이벤트 핸들러
// ────────────────────────────────────

// 허가 핀 상태별 스타일 생성
function makePermitPinStyle(status) {
    // 상태: 한글('승인'/'대기'/'거절') 또는 영어('approved'/'pending'/'rejected')
    const colorMap = {
        '승인': '#22c55e', 'approved': '#22c55e',
        '대기': '#38bdf8', 'pending':  '#38bdf8',
        '거절': '#ef4444', 'rejected': '#ef4444'
    };
    const emojiMap = {
        '승인': '✅', 'approved': '✅',
        '대기': '⏳', 'pending':  '⏳',
        '거절': '❌', 'rejected': '❌'
    };
    const color = colorMap[status] || '#94a3b8';
    const emoji = emojiMap[status] || '📍';

    // SVG 핀 아이콘 생성
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="36" viewBox="0 0 28 36">
    <path d="M14 0C6.27 0 0 6.27 0 14c0 9.75 14 22 14 22s14-12.25 14-22C28 6.27 21.73 0 14 0z"
        fill="${color}" stroke="white" stroke-width="2"/>
    <circle cx="14" cy="14" r="6" fill="white" opacity="0.9"/>
    </svg>`;
    const url = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);

    return new ol.style.Style({
        image: new ol.style.Icon({
            src: url,
            anchor: [0.5, 1.0],
            anchorXUnits: 'fraction',
            anchorYUnits: 'fraction',
            scale: 1.1
        }),
        text: new ol.style.Text({
            text: emoji,
            offsetY: -18,
            font: '13px sans-serif'
        })
    });
}

// 지도 아래 핀 정보 카드 표시 (핀 클릭 시 호출)
const STATUS_LABEL_HTML = {
    '승인': '<span style="color:#22c55e;font-weight:700;">● 승인</span>',
    '대기': '<span style="color:#38bdf8;font-weight:700;">● 대기 중</span>',
    '거절': '<span style="color:#ef4444;font-weight:700;">● 거절</span>',
    'approved': '<span style="color:#22c55e;font-weight:700;">● 승인</span>',
    'pending':  '<span style="color:#38bdf8;font-weight:700;">● 대기 중</span>',
    'rejected': '<span style="color:#ef4444;font-weight:700;">● 거절</span>'
};

function showPinCard(p, pixel) {
    const card = document.getElementById('pin-info-card');
    if (!card) return;

    const isRejected = (p.status === '거절' || p.status === 'rejected');

    document.getElementById('pin-applicant').textContent = p.user_name || '내 신청';
    document.getElementById('pin-status').innerHTML      = STATUS_LABEL_HTML[p.status] || p.status;
    document.getElementById('pin-purpose').textContent   = p.purpose || '-';
    document.getElementById('pin-period').textContent    = `${p.start_date} ~ ${p.end_date}`;
    document.getElementById('pin-radius').textContent    = `${p.radius}m`;
    document.getElementById('pin-coord').textContent     = `${parseFloat(p.lat).toFixed(4)}, ${parseFloat(p.lng).toFixed(4)}`;

    const rejectWrap = document.getElementById('pin-reject-wrap');
    if (isRejected) {
        document.getElementById('pin-reject-reason').textContent = p.reject_reason || '(사유 미입력)';
        rejectWrap.style.display = 'block';
    } else {
        rejectWrap.style.display = 'none';
    }

    // 일단 보이게 해서 크기 측정
    card.style.display = 'block';

    // 핀 클릭 픽셀 기준으로 카드 위치 결정
    if (pixel) {
        const mapEl = document.getElementById('vmap');
        const cardW = card.offsetWidth  || 260;
        const cardH = card.offsetHeight || 200;

        let left = pixel[0] + 15;   // 핀 오른쪽
        let top  = pixel[1] - 20;   // 핀 약간 위

        // 오른쪽 밖으로 넘치면 왼쪽에 표시
        if (left + cardW > mapEl.offsetWidth)  left = pixel[0] - cardW - 15;
        // 아래로 넘치면 위로 올림
        if (top  + cardH > mapEl.offsetHeight) top  = pixel[1] - cardH;
        // 음수 방지
        if (left < 0) left = 5;
        if (top  < 0) top  = 5;

        card.style.left = left + 'px';
        card.style.top  = top  + 'px';
    }
}

function closePinCard() {
    const card = document.getElementById('pin-info-card');
    if (card) card.style.display = 'none';
}

// 마우스 이동 → 좌표 실시간 표시 + 허가 핀 커서
function onPointerMove(e) {
    const [lng, lat] = ol.proj.toLonLat(e.coordinate);
    const latEl = document.getElementById('coord-lat');
    const lngEl = document.getElementById('coord-lng');
    if (latEl) latEl.textContent = lat.toFixed(5);
    if (lngEl) lngEl.textContent = lng.toFixed(5);

    // 허가 핀 위에서 pointer 커서
    const hit = vMap.hasFeatureAtPixel(e.pixel, {
        layerFilter: l => l === permitPinLayer
    });
    vMap.getTargetElement().style.cursor = hit ? 'pointer' : '';
}

// 지도 클릭 → 정보 패널 표시
function onMapClick(e) {
    const [lng, lat] = ol.proj.toLonLat(e.coordinate);
    const latStr = lat.toFixed(5);
    const lngStr = lng.toFixed(5);

    pinSource.clear();
    pinSource.addFeature(new ol.Feature({ geometry: new ol.geom.Point(e.coordinate) }));

    const zoneResult = showInfoPanel(latStr, lngStr);  // ← 반환값 받기

    if (typeof window.onMapClickWeather === 'function') {
        window.onMapClickWeather(latStr, lngStr, zoneResult.type);  // ← zoneType 전달
    }
}


// ────────────────────────────────────
// 5. 구역 데이터 로드 (Flask API 연동)
//    수업 패턴: async/await + fetch
// ────────────────────────────────────
async function loadZoneData() {
    try {
        const res  = await fetch('/api/zones');
        const data = await res.json();

        if (!data.ok) {
            console.error('구역 데이터 오류:', data.msg);
            loadFallbackZones();
            return;
        }

        data.zones.forEach(zone => addZoneFeature(zone));
        console.log(`구역 데이터 로드 완료: ${data.zones.length}개`);

    } catch (err) {
        console.error('구역 API 호출 실패:', err);
        loadFallbackZones();
    }
}

// 폴리곤 피처 추가
// zone: { id, name, code, type, category, coords:[[lng,lat],...] }
function addZoneFeature(zone) {
    const type = zone.type;
    if (!zoneSources[type]) return;  // 알 수 없는 타입 무시

    const ring    = [zone.coords.map(c => ol.proj.fromLonLat(c))];
    const feature = new ol.Feature({
        geometry:  new ol.geom.Polygon(ring),
        zone_name: zone.name,
        zone_code: zone.code,
        zone_type: type
    });
    zoneSources[type].addFeature(feature);
}

// API 실패 시 개발용 예시 데이터
function loadFallbackZones() {
    console.warn('fallback 예시 데이터로 표시합니다.');
    const samples = [
        { type: 'forbidden',  name: '서울 도심 비행금지',  code: 'P61A',
            coords: [[126.97,37.57],[126.99,37.57],[126.99,37.55],[126.97,37.55],[126.97,37.57]] },
        { type: 'restricted', name: '김포공항 비행제한',    code: 'R125',
            coords: [[126.43,37.47],[126.47,37.47],[126.47,37.44],[126.43,37.44],[126.43,37.47]] },
        { type: 'danger',     name: '부산 비행위험구역',    code: 'D123',
            coords: [[129.00,35.20],[129.04,35.20],[129.04,35.17],[129.00,35.17],[129.00,35.20]] },
        { type: 'permit',     name: '강남구 허가요청구역',  code: 'PERMIT-001',
            coords: [[127.10,37.40],[127.14,37.40],[127.14,37.37],[127.10,37.37],[127.10,37.40]] }
    ];
    samples.forEach(zone => addZoneFeature(zone));
}


// ────────────────────────────────────
// 6. 정보 패널
// ────────────────────────────────────
function showInfoPanel(lat, lng) {
    const result = checkZoneAtPoint(lat, lng);

    document.getElementById('info-lat').textContent  = lat;
    document.getElementById('info-lng').textContent  = lng;
    document.getElementById('info-zone').textContent = result.label;
    document.getElementById('info-name').textContent = result.name;
    document.getElementById('info-addr').textContent = '조회 중...';
    document.getElementById('map-info-panel').classList.add('show');

    fetchAddress(lat, lng);
    saveSearchLog(lat, lng);

    return result;  // ← 추가
}

function closeInfoPanel() {
    document.getElementById('map-info-panel').classList.remove('show');
}

// 클릭 좌표 → 구역 판별 (OpenLayers intersectsCoordinate 사용)
function checkZoneAtPoint(lat, lng) {
    const coord = ol.proj.fromLonLat([parseFloat(lng), parseFloat(lat)]);
    for (const type of ['forbidden', 'restricted', 'danger', 'permit']) {
        const features = zoneSources[type].getFeatures();
        const hit = features.find(f => f.getGeometry().intersectsCoordinate(coord));
        if (hit) {
            return { label: ZONE_LABEL[type], name: hit.get('zone_name') || '-', type: type };
        }
    }
    return { label: '✈️ 비행 가능', name: '-', type: 'normal' };
}

// ────────────────────────────────────
// 7. vworld 주소 역지오코딩
//    수업 패턴: async/await fetch
// ────────────────────────────────────
async function fetchAddress(lat, lng) {
    const url = `/api/address?lat=${lat}&lng=${lng}`;
    try {
        const res  = await fetch(url);
        const data = await res.json();

        if (!data.ok) {
            // ── DB에 없는 지역 → 경고창 ──────────────────────
            document.getElementById('info-addr').textContent = '등록되지 않은 지역';
            updateLocationCard(null, lat, lng);
            alert(
                `⚠️ 선택한 위치가 행정구역 DB에 없습니다.\n\n` +
                `위도: ${lat}  경도: ${lng}\n\n` +
                `한국 육지 지역을 선택해주세요.\n(해상, 도서 지역은 조회되지 않을 수 있습니다.)`
            );
            return;
        }

        const addr = data.address;
        document.getElementById('info-addr').textContent = addr;
        updateLocationCard(addr, lat, lng);

    } catch(e) {
        document.getElementById('info-addr').textContent = '조회 실패';
        updateLocationCard(null, lat, lng);
    }
}

function updateLocationCard(addr, lat, lng) {
    const card = document.getElementById('panel-location');
    if (!card) return;
    card.style.display = 'block';

    // "수원시영통구" → "수원시 영통구" 자동 띄어쓰기
    function formatSigungu(s) {
        if (!s) return '';
        return s.replace(/(시|군)([가-힣]+[구읍면동])/, '$1 $2');
    }

    if (!addr) {
        document.getElementById('loc-province').textContent = '조회 실패';
        document.getElementById('loc-city').textContent     = '';
        document.getElementById('loc-dong').textContent     = '';
    } else {
        // 예: "경기도 남양주시 와부읍 팔당리 123"
        const parts = addr.trim().split(/\s+/);
        document.getElementById('loc-province').textContent = parts[0] || '-';
        document.getElementById('loc-city').textContent     = formatSigungu(parts[1]) || '';  // ← 여기!
        document.getElementById('loc-dong').textContent     = parts.slice(2, 4).join(' ') || '';
    }

    document.getElementById('loc-lat').textContent = `위도 ${parseFloat(lat).toFixed(4)}`;
    document.getElementById('loc-lng').textContent = `경도 ${parseFloat(lng).toFixed(4)}`;
}

// ────────────────────────────────────
// 8. 내 허가 신청 현황 패널 렌더링
// ────────────────────────────────────
function renderPermitStatusPanel(permits) {
    const list = document.getElementById('psp-list');
    if (!list) return;

    if (!permits || permits.length === 0) {
        list.innerHTML = '<span class="psp-empty">신청 내역이 없습니다.</span>';
        return;
    }

    const statusKr  = { '승인': '승인', '대기': '대기 중', '거절': '거절' };
    const statusCls = { '승인': 'approved', '대기': 'pending', '거절': 'rejected' };

    list.innerHTML = permits.map((p, i) => {
        const cls   = statusCls[p.status] || 'pending';
        const label = statusKr[p.status]  || p.status;
        const displayNum = permits.length - i;  // 내림차순 순번

        const rejectBox = (p.status === '거절' && p.reject_reason)
            ? `<div class="psp-reject-box">
                    <span class="psp-reject-label">거절 사유</span>
                    ${p.reject_reason}
                </div>`
            : '';

        const latStr = parseFloat(p.lat).toFixed(4);
        const lngStr = parseFloat(p.lng).toFixed(4);

        return `
        <div class="psp-card" data-id="${p.id}" onclick="openPspDetailModal(${JSON.stringify(p).replace(/"/g,'&quot;')})" style="cursor:pointer;">
            <div class="psp-card-head ${cls}">
                <span>
                    <span class="psp-status-dot ${cls}"></span>
                    <span class="psp-status-text ${cls}">${label}</span>
                </span>
                <span class="psp-req-id">#${displayNum}</span>
            </div>
            <div class="psp-card-body">
                <div class="psp-row">
                    <span class="psp-label">요청</span>
                    <span class="psp-value" id="psp-addr-${p.id}">${latStr}, ${lngStr}</span>
                </div>
                <div class="psp-row">
                    <span class="psp-label">목적</span>
                    <span class="psp-value">${p.purpose || '-'}</span>
                </div>
                <div class="psp-row">
                    <span class="psp-label">기간</span>
                    <span class="psp-value" style="font-size:0.68rem;">${p.start_date} ~ ${p.end_date}</span>
                </div>
                ${rejectBox}
            </div>
        </div>`;
    }).join('');

    // 좌표 → 주소 역지오코딩 (비동기 후처리, 실패해도 좌표 유지)
    permits.forEach(p => {
        fetch(`/api/address?lat=${p.lat}&lng=${p.lng}`)
            .then(r => r.json())
            .then(d => {
                const el = document.getElementById(`psp-addr-${p.id}`);
                if (el && d.ok) el.textContent = d.address;
            })
            .catch(() => {});
    });
}


// ────────────────────────────────────
// 내 허가 신청 구역 로드 (일반 유저)
// ────────────────────────────────────
async function loadUserPermits() {
    try {
        const res  = await fetch('/api/my_permits');
        const data = await res.json();

        // 패널 렌더링 (신청 없어도 "내역 없음" 표시)
        renderPermitStatusPanel(data.permits || []);

        if (!data.ok || !data.permits || data.permits.length === 0) {
            hasUserPermits = false;
            const btn = document.getElementById('my-permit-btn');
            if (btn) {
                btn.classList.remove('active');
                btn.style.opacity = '0.45';
            }
            return;
        }
        hasUserPermits = true;

        const colorMap = {
            '승인': { fill: 'rgba(34,197,94,0.18)',  stroke: 'rgba(34,197,94,0.9)'  },
            '거절': { fill: 'rgba(239,68,68,0.10)',  stroke: 'rgba(239,68,68,0.75)' },
            '대기': { fill: 'rgba(56,189,248,0.12)', stroke: 'rgba(56,189,248,0.9)' }
        };

        data.permits.forEach(p => {
            if (!p.lat || !p.lng) return;
            const coord = ol.proj.fromLonLat([p.lng, p.lat]);
            const pRes  = ol.proj.getPointResolution('EPSG:3857', 1, coord);
            const geom  = ol.geom.Polygon.fromCircle(
                new ol.geom.Circle(coord, (p.radius || 500) / pRes)
            );
            const c    = colorMap[p.status] || colorMap['대기'];
            const feat = new ol.Feature(geom);
            feat.setStyle(new ol.style.Style({
                fill:   new ol.style.Fill({ color: c.fill }),
                stroke: new ol.style.Stroke({ color: c.stroke, width: 2, lineDash: [6, 4] })
            }));
            userPermitSource.addFeature(feat);

            // ── 핀 마커 추가 ──────────────────────────────
            const pinFeat = new ol.Feature({ geometry: new ol.geom.Point(coord) });
            pinFeat.setStyle(makePermitPinStyle(p.status));
            pinFeat.set('_permit', p);   // 팝업용 데이터 저장
            permitPinSource.addFeature(pinFeat);
        });
        console.log(`내 허가 신청 구역 ${data.permits.length}건 표시`);

    } catch(e) {
        console.warn('내 허가 구역 로드 실패:', e);
    }
}

// 내 허가 신청 토글 (버튼 클릭)
function toggleUserPermitLayer(btn) {
    if (!hasUserPermits) {
        alert('비행 승인 요청 이력이 없습니다!');
        return;
    }
    const visible = userPermitLayer.getVisible();
    const nextVisible = !visible;

    // 바운더리 레이어 토글
    userPermitLayer.setVisible(nextVisible);

    // 핀 마커 토글 (모든 핀 — 유저 핀은 전부 본인 것)
    permitPinSource.getFeatures().forEach(f => {
        f.setStyle(nextVisible ? makePermitPinStyle(f.get('_permit').status) : new ol.style.Style({}));
    });

    btn.classList.toggle('active', nextVisible);
}


// ────────────────────────────────────
// 9. 관리자 전체 허가 신청 로드
// ────────────────────────────────────
async function loadAllPermits(dateFrom = '', dateTo = '') {
    try {
        // 기존 데이터 초기화
        Object.values(adminPermitSources).forEach(s => s.clear());
        permitPinSource.getFeatures()
            .filter(f => f.get('_permit') && f.get('_permit').user_name !== undefined)
            .forEach(f => permitPinSource.removeFeature(f));

        const params = new URLSearchParams();
        if (dateFrom) params.set('date_from', dateFrom);
        if (dateTo)   params.set('date_to',   dateTo);
        const url = '/api/all_permits' + (params.toString() ? '?' + params.toString() : '');

        const res  = await fetch(url);
        const data = await res.json();
        if (!data.ok || !data.permits || data.permits.length === 0) return;

        data.permits.forEach(p => {
            if (!p.lat || !p.lng) return;
            const status = p.status;
            if (!adminPermitSources[status]) return;

            const coord = ol.proj.fromLonLat([p.lng, p.lat]);
            const pRes  = ol.proj.getPointResolution('EPSG:3857', 1, coord);
            const geom  = ol.geom.Polygon.fromCircle(
                new ol.geom.Circle(coord, (p.radius || 500) / pRes)
            );
            const feat = new ol.Feature(geom);
            adminPermitSources[status].addFeature(feat);

            // ── 핀 마커 추가 ──────────────────────────────
            const pinFeat = new ol.Feature({ geometry: new ol.geom.Point(coord) });
            pinFeat.setStyle(makePermitPinStyle(status));
            pinFeat.set('_permit', p);
            permitPinSource.addFeature(pinFeat);
        });
        console.log(`관리자 허가 현황 ${data.permits.length}건 로드`);
    } catch(e) {
        console.warn('관리자 허가 로드 실패:', e);
    }
}

// 날짜 필터 적용
function applyAdminDateFilter() {
    const dateFrom = document.getElementById('admin-date-from')?.value || '';
    const dateTo   = document.getElementById('admin-date-to')?.value   || '';

    if (dateFrom && dateTo && dateFrom > dateTo) {
        // 에러 표시: 입력창 테두리 빨간색 + 알림
        const fromEl = document.getElementById('admin-date-from');
        const toEl   = document.getElementById('admin-date-to');
        fromEl.style.borderColor = '#ef4444';
        toEl.style.borderColor   = '#ef4444';

        // 기존 에러 메시지 제거 후 새로 표시
        const existing = document.getElementById('admin-date-error');
        if (existing) existing.remove();
        const errEl = document.createElement('span');
        errEl.id = 'admin-date-error';
        errEl.textContent = '⚠️ 시작 날짜는 종료 날짜보다 이전이어야 합니다.';
        errEl.style.cssText = 'font-size:0.7rem; color:#ef4444; white-space:nowrap;';
        toEl.parentNode.insertBefore(errEl, toEl.nextSibling);

        setTimeout(() => {
            fromEl.style.borderColor = '#1e2d45';
            toEl.style.borderColor   = '#1e2d45';
            errEl.remove();
        }, 3000);
        return;
    }

    // 에러 초기화
    const existing = document.getElementById('admin-date-error');
    if (existing) existing.remove();

    loadAllPermits(dateFrom, dateTo);
}

// 날짜 필터 초기화 (오늘~+7일로 리셋)
function resetAdminDateFilter() {
    const fromEl = document.getElementById('admin-date-from');
    const toEl   = document.getElementById('admin-date-to');
    const today  = getDateStr(0);
    const week   = getDateStr(7);
    if (fromEl) fromEl.value = today;
    if (toEl)   toEl.value   = week;
    loadAllPermits(today, week);
}

// 상태별 레이어 토글 (관리자)
function toggleAllPermitByStatus(status, btn) {
    const layer = adminPermitLayers[status];
    if (!layer) return;
    const visible    = layer.getVisible();
    const nextVisible = !visible;

    // 바운더리 레이어 토글
    layer.setVisible(nextVisible);

    // 해당 상태의 핀 마커만 토글
    permitPinSource.getFeatures().forEach(f => {
        const p = f.get('_permit');
        if (p && p.status === status) {
            f.setStyle(nextVisible ? makePermitPinStyle(status) : new ol.style.Style({}));
        }
    });

    btn.classList.toggle('active', nextVisible);
}


// ────────────────────────────────────
// 9. 조회 기록 저장 (Flask API POST)
//    수업 패턴: fetch POST + JSON
// ────────────────────────────────────
async function saveSearchLog(lat, lng) {
    try {
        await fetch('/api/zone_log', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ latitude: lat, longitude: lng })
        });
    } catch (err) {
        console.warn('조회 기록 저장 실패:', err);
    }
}


// ────────────────────────────────────
// 10. 레이어 토글
// ────────────────────────────────────
function toggleLayer(type, btn) {
    layerVisible[type] = !layerVisible[type];
    btn.classList.toggle('active', layerVisible[type]);
    zoneLayers[type].setVisible(layerVisible[type]);
}


// ────────────────────────────────────
// 11. 줌 컨트롤
// ────────────────────────────────────
function mapZoomIn()  { vMap.getView().setZoom(vMap.getView().getZoom() + 1); }
function mapZoomOut() { vMap.getView().setZoom(vMap.getView().getZoom() - 1); }
function mapReset()   {
    vMap.getView().setCenter(KR_CENTER);
    vMap.getView().setZoom(INIT_ZOOM);
}


// ────────────────────────────────────
// 12. 진입점 (DOMContentLoaded)
// ────────────────────────────────────

// 날짜 input 연도 4자리 제한
function limitYear(input) {
    if (!input.value) return;
    const parts = input.value.split('-');
    if (parts[0] && parts[0].length > 4) {
        parts[0] = parts[0].slice(0, 4);
        input.value = parts.join('-');
    }
}

// 오늘 기준 날짜 문자열 반환 (±days)
function getDateStr(offsetDays = 0) {
    const d = new Date();
    d.setDate(d.getDate() + offsetDays);
    return d.toISOString().slice(0, 10);
}

document.addEventListener('DOMContentLoaded', () => {
    initMap();       // 지도 초기화
    loadZoneData();  // 구역 데이터 로드

    // 지도 컨테이너 크기 재계산 (CSS flex 레이아웃 안정화 후)
    setTimeout(() => { if (vMap) vMap.updateSize(); }, 100);
    setTimeout(() => { if (vMap) vMap.updateSize(); }, 500);

    // ── 기본 위치 핀 + 정보 패널 표시 ──────────────────────
    //    weather_section.js 의 DEFAULT_LAT / DEFAULT_LON 사용
    //    (두 파일 모두 같은 좌표를 공유)
    const defLat = typeof DEFAULT_LAT !== 'undefined' ? DEFAULT_LAT : 37.49985;
    const defLon = typeof DEFAULT_LON !== 'undefined' ? DEFAULT_LON : 127.03383;
    const defLatStr = defLat.toFixed(5);
    const defLonStr = defLon.toFixed(5);

    // 핀 마커를 기본 위치에 표시
    const defCoord = ol.proj.fromLonLat([defLon, defLat]);
    pinSource.clear();
    pinSource.addFeature(new ol.Feature({ geometry: new ol.geom.Point(defCoord) }));

    // 위치 정보 패널 + 선택 위치 카드 표시
    showInfoPanel(defLatStr, defLonStr);

    // ── 로그인 상태별 허가 구역 로드 ──────────────────────
    if (typeof IS_ADMIN !== 'undefined' && IS_ADMIN) {
        // 기본값: 오늘 ~ 오늘+7일
        const fromEl = document.getElementById('admin-date-from');
        const toEl   = document.getElementById('admin-date-to');
        const today  = getDateStr(0);
        const week   = getDateStr(7);
        if (fromEl) fromEl.value = today;
        if (toEl)   toEl.value   = week;
        loadAllPermits(today, week);   // 관리자: 기본 1주일 신청 현황
    } else if (typeof IS_LOGGED_IN !== 'undefined' && IS_LOGGED_IN) {
        loadUserPermits();             // 일반 유저: 내 신청 현황
    }
});


// ────────────────────────────────────────────────────────────
// 허가 신청 상세/수정 모달 (psp-detail-modal)
// ────────────────────────────────────────────────────────────
const PSP_STATUS_STYLE = {
    '승인': { text: '✅ 승인',   bg: 'rgba(22,163,74,0.15)',  color: '#22c55e' },
    '대기': { text: '⏳ 대기 중', bg: 'rgba(56,189,248,0.15)', color: '#38bdf8' },
    '거절': { text: '❌ 거절',   bg: 'rgba(220,38,38,0.15)',  color: '#ef4444' },
};

function openPspDetailModal(p) {
    const modal = document.getElementById('psp-detail-modal');
    if (!modal) return;

    // 기본 데이터 주입
    document.getElementById('psp-d-id').value = p.id;

    // 상태 배지
    const st = PSP_STATUS_STYLE[p.status] || { text: p.status, bg: '#334155', color: '#94a3b8' };
    const badge = document.getElementById('psp-d-status-badge');
    badge.textContent   = st.text;
    badge.style.background = st.bg;
    badge.style.color      = st.color;

    const isPending = (p.status === '대기');
    document.getElementById('psp-d-readonly').style.display = isPending ? 'none' : 'block';
    document.getElementById('psp-d-edit').style.display     = isPending ? 'block' : 'none';

    if (isPending) {
        // 수정 폼 채우기
        document.getElementById('psp-d-purpose').value  = p.purpose  || '';
        document.getElementById('psp-d-drone').value    = p.drone_type || '';
        document.getElementById('psp-d-altitude').value = p.flight_altitude || '';
        const editPhotoEl = document.getElementById('psp-d-photo-text');
        if (editPhotoEl) editPhotoEl.textContent = p.photo_request ? '✅ 촬영 신청함' : '🚫 촬영 안함';
        document.getElementById('psp-d-start').value    = p.start_date || '';
        document.getElementById('psp-d-end').value      = p.end_date   || '';
        document.getElementById('psp-d-lat').value      = p.lat ? parseFloat(p.lat).toFixed(6) : '';
        document.getElementById('psp-d-lng').value      = p.lng ? parseFloat(p.lng).toFixed(6) : '';
        document.getElementById('psp-d-radius').value   = p.radius || 500;
        // 첨부파일 목록 (수정 폼용)
        loadPspFiles(p.id, 'psp-d-files-list-edit');
    } else {
        // 읽기 전용 데이터 채우기
        document.getElementById('psp-ro-purpose').textContent  = p.purpose  || '-';
        document.getElementById('psp-ro-drone').textContent    = p.drone_type || '-';
        document.getElementById('psp-ro-period').textContent   = `${p.start_date} ~ ${p.end_date}`;
        document.getElementById('psp-ro-coord').textContent    =
            p.lat ? `${parseFloat(p.lat).toFixed(5)}, ${parseFloat(p.lng).toFixed(5)}` : '-';
        document.getElementById('psp-ro-radius').textContent   = p.radius || 500;
        document.getElementById('psp-ro-altitude').textContent = p.flight_altitude || '-';
        const roPhotoEl = document.getElementById('psp-ro-photo');
        if (roPhotoEl) roPhotoEl.textContent = p.photo_request ? '✅ 촬영 신청함' : '🚫 촬영 안함';

        // 거절 사유
        const rjWrap = document.getElementById('psp-d-reject-wrap');
        if (p.status === '거절' && p.reject_reason) {
            document.getElementById('psp-d-reject-reason').textContent = p.reject_reason;
            rjWrap.style.display = 'block';
        } else {
            rjWrap.style.display = 'none';
        }
        // 첨부파일 목록 (읽기 전용)
        loadPspFiles(p.id, 'psp-d-files-list');
    }

    modal.style.display = 'flex';
}

function closePspDetailModal() {
    const modal = document.getElementById('psp-detail-modal');
    if (modal) modal.style.display = 'none';
}

// 첨부파일 목록 불러오기 (모달 내 표시)
async function loadPspFiles(reqId, targetElId) {
    const el = document.getElementById(targetElId);
    if (!el) return;
    el.innerHTML = '<span style="color:var(--text-muted); font-size:0.78rem;">불러오는 중...</span>';
    try {
        const res  = await fetch(`/api/my_permits/${reqId}/files`);
        const data = await res.json();
        if (!data.ok || data.files.length === 0) {
            el.innerHTML = '<span style="color:var(--text-muted); font-size:0.78rem;">첨부파일 없음</span>';
            return;
        }
        el.innerHTML = data.files.map(f => `
            <div style="display:flex; align-items:center; gap:0.5rem; padding:0.3rem 0;
                        border-bottom:1px solid var(--border); font-size:0.8rem;">
                <span>📄</span>
                <span style="flex:1; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    ${f.original_name}
                </span>
                <span style="color:var(--text-muted); font-size:0.72rem; white-space:nowrap;">
                    ${f.file_size ? (f.file_size/1024).toFixed(1)+' KB' : ''}
                </span>
                <a href="${f.download_url}" download="${f.original_name}"
                    style="padding:0.18rem 0.55rem; background:var(--surface2); border:1px solid var(--border);
                            border-radius:5px; font-size:0.72rem; color:var(--accent2); text-decoration:none;
                            white-space:nowrap;"
                    onmouseover="this.style.background='var(--accent)';this.style.color='#fff';"
                    onmouseout="this.style.background='var(--surface2)';this.style.color='var(--accent2)';">
                    ⬇ 다운로드
                </a>
            </div>
        `).join('');
    } catch {
        el.innerHTML = '<span style="color:var(--text-muted); font-size:0.78rem;">파일 목록 조회 실패</span>';
    }
}

// 수정 저장
async function submitPspUpdate() {
    const id = document.getElementById('psp-d-id').value;
    const body = {
        purpose:          document.getElementById('psp-d-purpose').value.trim(),
        drone_type:       document.getElementById('psp-d-drone').value.trim(),
        flight_altitude:  parseInt(document.getElementById('psp-d-altitude').value) || null,
        start_date:       document.getElementById('psp-d-start').value,
        end_date:         document.getElementById('psp-d-end').value,
        latitude:         parseFloat(document.getElementById('psp-d-lat').value) || null,
        longitude:        parseFloat(document.getElementById('psp-d-lng').value) || null,
        radius:           parseInt(document.getElementById('psp-d-radius').value) || 500,
    };
    if (!body.purpose)              { alert('사용 목적을 입력해주세요.'); return; }
    if (!body.start_date || !body.end_date) { alert('비행 기간을 입력해주세요.'); return; }

    try {
        const res  = await fetch(`/api/my_permits/${id}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.ok) {
            alert('✅ 수정이 완료되었습니다.');
            closePspDetailModal();
            loadUserPermits();   // 카드 목록 새로고침
        } else {
            alert('❌ ' + data.msg);
        }
    } catch (e) {
        alert('오류: ' + e.message);
    }
}