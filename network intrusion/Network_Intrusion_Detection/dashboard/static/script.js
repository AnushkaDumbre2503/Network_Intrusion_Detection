document.addEventListener('DOMContentLoaded', () => {

    const fileInput = document.getElementById('csvFile');
    const fileNameDisplay = document.getElementById('fileName');
    const runAnalysisBtn = document.getElementById('runAnalysisBtn');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const dashboardContent = document.getElementById('dashboardContent');
    const visualsContent = document.getElementById('visualsContent');
    const statsContent = document.getElementById('statsContent');
    const insightsContent = document.getElementById('insightsContent');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const systemStatus = document.getElementById('systemStatus');
    const viewResultsBtn = document.getElementById('viewResultsBtn');
    const navDashboard = document.getElementById('navDashboard');
    const navVisuals = document.getElementById('navVisuals');
    const navStats = document.getElementById('navStats');
    const navInsights = document.getElementById('navInsights');

    const metricTotalFlows = document.getElementById('metricTotalFlows');
    const metricAnomalies = document.getElementById('metricAnomalies');
    const metricHighRisk = document.getElementById('metricHighRisk');

    const API_BASE = '';

    let selectedFile = null;

    // ---------------- FILE SELECT ----------------
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            fileNameDisplay.textContent = selectedFile.name;
            fileNameDisplay.style.color = 'var(--accent-primary)';
            runAnalysisBtn.disabled = false;
        } else {
            selectedFile = null;
            fileNameDisplay.textContent = 'No file selected';
        }
    });

    // ---------------- RUN ANALYSIS ----------------
    runAnalysisBtn.addEventListener('click', async () => {

        if (!selectedFile) {
            alert('Please select a CSV file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        systemStatus.textContent = 'Processing Data...';

        setActiveNav('navDashboard');
        welcomeScreen.classList.add('hidden');
        visualsContent.classList.add('hidden');
        statsContent.classList.add('hidden');
        insightsContent.classList.add('hidden');
        dashboardContent.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');

        try {
            const response = await fetch(`${API_BASE}/upload-data`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data || data.error || data.detail) {
                console.error("Backend error:", data);
                alert(data?.error || data?.detail || "Server error");
                loadingOverlay.classList.add('hidden');
                return;
            }

            setTimeout(() => {
                console.log("RAW API RESPONSE:", data);
                renderDashboard(data);
                loadingOverlay.classList.add('hidden');
                systemStatus.textContent = 'Analysis Complete';
            }, 500);

        } catch (err) {
            console.error(err);
            alert("Request failed");
            loadingOverlay.classList.add('hidden');
        }
    });

    // ---------------- VIEW RESULTS ----------------
    viewResultsBtn.addEventListener('click', async (e) => {
        e.preventDefault();

        systemStatus.textContent = 'Fetching Results...';

        setActiveNav('viewResultsBtn');
        welcomeScreen.classList.add('hidden');
        visualsContent.classList.add('hidden');
        statsContent.classList.add('hidden');
        insightsContent.classList.add('hidden');
        dashboardContent.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');

        try {
            const response = await fetch(`${API_BASE}/results`);
            const data = await response.json();
            console.log("API RESPONSE:", data);
            setTimeout(() => {

                if (!data || !data.summary || data.summary.total_flows === 0) {
                    alert('No previous analysis found.');
                    dashboardContent.classList.add('hidden');
                    welcomeScreen.classList.remove('hidden');
                } else {
                    console.log("RAW API RESPONSE:", data);
                    renderDashboard(data);
                }

                loadingOverlay.classList.add('hidden');

            }, 500);

        } catch (err) {
            console.error(err);
            loadingOverlay.classList.add('hidden');
        }
    });

    // ---------------- NAVIGATION TABS ----------------
    function setActiveNav(activeId) {
        document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
        if (activeId) {
            const el = document.getElementById(activeId);
            if (el && el.parentElement) el.parentElement.classList.add('active');
        }
    }

    navDashboard.addEventListener('click', (e) => {
        e.preventDefault();
        setActiveNav('navDashboard');
        welcomeScreen.classList.add('hidden');
        visualsContent.classList.add('hidden');
        statsContent.classList.add('hidden');
        insightsContent.classList.add('hidden');
        dashboardContent.classList.remove('hidden');
    });

    navVisuals.addEventListener('click', (e) => {
        e.preventDefault();
        setActiveNav('navVisuals');
        welcomeScreen.classList.add('hidden');
        dashboardContent.classList.add('hidden');
        statsContent.classList.add('hidden');
        insightsContent.classList.add('hidden');
        visualsContent.classList.remove('hidden');
    });

    navStats.addEventListener('click', async (e) => {
        e.preventDefault();
        setActiveNav('navStats');
        welcomeScreen.classList.add('hidden');
        dashboardContent.classList.add('hidden');
        visualsContent.classList.add('hidden');
        insightsContent.classList.add('hidden');
        statsContent.classList.remove('hidden');

        await fetchStatsIfNeeded();
    });

    navInsights.addEventListener('click', async (e) => {
        e.preventDefault();
        setActiveNav('navInsights');
        welcomeScreen.classList.add('hidden');
        dashboardContent.classList.add('hidden');
        visualsContent.classList.add('hidden');
        statsContent.classList.add('hidden');
        insightsContent.classList.remove('hidden');

        await fetchStatsIfNeeded();
    });

    async function fetchStatsIfNeeded() {
        const statsGrid = document.getElementById('statsGrid');
        if (statsGrid && statsGrid.children.length === 0) {
            try {
                systemStatus.textContent = 'Fetching Stats...';
                loadingOverlay.classList.remove('hidden');
                
                const res = await fetch('data/eda_stats.json');
                const statsData = await res.json();
                renderStats(statsData);
                
                loadingOverlay.classList.add('hidden');
                systemStatus.textContent = 'System Ready';
            } catch (err) {
                console.error(err);
                loadingOverlay.classList.add('hidden');
                systemStatus.textContent = 'Failed to Load Stats';
            }
        }
    }

    // ---------------- STATS RENDER ----------------
    function renderStats(data) {
        const grid = document.getElementById('statsGrid');
        grid.replaceChildren();

        function addCard(title, entries, formatter = value => String(value)) {
            const card = document.createElement('div');
            card.className = 'stat-card glass-panel';
            const heading = document.createElement('h4');
            heading.textContent = title;
            card.appendChild(heading);
            const list = document.createElement('ul');
            list.className = 'stat-list';
            for (const [key, value] of Object.entries(entries || {})) {
                const item = document.createElement('li');
                item.textContent = `${key}: ${formatter(value, key)}`;
                list.appendChild(item);
            }
            card.appendChild(list);
            grid.appendChild(card);
        }

        const shape = data.data_overview?.shape || [0, 0];
        addCard('Data Overview', {Rows: shape[0] || 0, Columns: shape[1] || 0});
        addCard('Class Distribution', data.class_distribution, (value, key) =>
            `${Number(value).toLocaleString()} (${Number(data.class_percentage?.[key] || 0).toFixed(2)}%)`);
        addCard('Top Suspicious Ports', data.top_suspicious_ports, value => `${(Number(value) * 100).toFixed(2)}% attack ratio`);
        addCard('Most Used Destination Ports', data.top_ports, value => `${Number(value).toLocaleString()} flows`);
        addCard('Attack Types', data.attack_type, value => `${Number(value).toLocaleString()} flows`);

        const traffic = {};
        for (const [label, values] of Object.entries(data.traffic_intensity || {})) {
            traffic[label] = `Bytes/s ${Number(values['Flow Bytes/s'] || 0).toLocaleString(undefined, {maximumFractionDigits: 1})}; Packets/s ${Number(values['Flow Packets/s'] || 0).toLocaleString(undefined, {maximumFractionDigits: 1})}`;
        }
        addCard('Traffic Intensity', traffic);
        addCard('Traffic Imbalance', data.packet_imbalance, value => Number(value).toFixed(4));

        const timing = {};
        for (const [label, values] of Object.entries(data.timing_behavior || {})) {
            timing[label] = `IAT mean: ${Number(values['Flow IAT Mean'] || 0).toLocaleString(undefined, {maximumFractionDigits: 2})} µs; variability: ${Number(values.iat_variability || 0).toFixed(4)}`;
        }
        addCard('Timing Behaviour', timing);

        const flags = {};
        for (const [label, values] of Object.entries(data.flag_analysis || {})) {
            flags[label] = `SYN/ACK ratio: ${Number(values.flag_ratio || 0).toFixed(4)}; SYN: ${Number(values['SYN Flag Count'] || 0).toFixed(2)}; ACK: ${Number(values['ACK Flag Count'] || 0).toFixed(2)}`;
        }
        addCard('TCP Flag Statistics', flags);

        const dayStats = {};
        for (const [day, values] of Object.entries(data.day_wise_analysis || {})) {
            dayStats[day] = Object.values(values).reduce((sum, count) => sum + Number(count || 0), 0);
        }
        addCard('Day-wise Flow Counts', dayStats, value => Number(value).toLocaleString());

        const insightsList = document.getElementById('insightsList');
        insightsList.replaceChildren();
        (data.insights || []).forEach(insight => {
            const item = document.createElement('li');
            item.textContent = String(insight);
            item.style.marginBottom = '0.5rem';
            insightsList.appendChild(item);
        });
    }

    // ---------------- DASHBOARD RENDER ----------------
    function renderDashboard(data) {
        console.log("INSIDE DASHBOARD:", data);

        // 🔥 SAFETY CHECK (MOST IMPORTANT)
        if (!data || !data.summary) {
            console.error("Invalid data:", data);
            alert("Invalid response from server");
            return;
        }

        const summary = data.summary || {};
        const features = data.feature_importance || [];
        const temporal = data.temporal_data || {};
        const table = data.table_data || [];

        animateValue(metricTotalFlows, 0, summary.total_flows || 0, 800);
        animateValue(metricAnomalies, 0, summary.total_anomalies || 0, 800);
        animateValue(metricHighRisk, 0, summary.high_risk || 0, 800);

        // ---------------- FEATURE IMPORTANCE ----------------
        const featureList = document.getElementById('featureImportanceList');
        featureList.innerHTML = '';

        features.forEach(item => {
            const pct = ((item?.importance || 0) * 100).toFixed(1);

            featureList.insertAdjacentHTML('beforeend', `
                <div class="feature-item">
                    <span class="feature-name">${formatLabel(item?.feature || "")}</span>
                    <div class="feature-bar-bg">
                        <div class="feature-bar-fill" style="width:${pct}%"></div>
                    </div>
                    <span class="feature-value">${pct}%</span>
                </div>
            `);
        });

        // ---------------- TEMPORAL DATA ----------------
        const timelineBars = document.getElementById('timelineBars');
        timelineBars.innerHTML = '';

        const times = temporal.time_bucket || [];
        const counts = temporal.anomaly_count || [];
        const max = Math.max(...counts, 0);
        if (temporal.available === false && temporal.message) {
            const note = document.createElement('p');
            note.className = 'muted';
            note.textContent = temporal.message;
            timelineBars.appendChild(note);
        }

        times.forEach((t, i) => {
            const count = counts[i] || 0;
            const h = max ? (count / max) * 100 : 0;

            timelineBars.insertAdjacentHTML('beforeend', `
                <div class="timeline-bar-container">
                    <div class="t-bar" style="height:${h}%"></div>
                    <span class="t-label">${t}</span>
                </div>
            `);
        });

        // ---------------- TABLE ----------------
        const tbody = document.getElementById('tableBody');
        tbody.innerHTML = '';

        // Uploaded CSV values are untrusted: build cells with textContent, never HTML interpolation.
        table.forEach(row => {
            const tr = document.createElement('tr');
            const values = [
                `#${row.flow_id ?? ""}`,
                row.attack_type ?? "",
                row.risk_level ?? "Low",
                Number(row.risk_score ?? 0).toFixed(3),
                Number(row.anomaly_score ?? 0).toFixed(6),
                `${(Number(row.confidence ?? 0) * 100).toFixed(1)}%`,
                row.behavior_tag ?? "",
                row.explanation ?? ""
            ];
            values.forEach((value, index) => {
                const td = document.createElement('td');
                if (index === 2) {
                    const badge = document.createElement('span');
                    badge.className = 'badge';
                    badge.textContent = String(value);
                    td.appendChild(badge);
                } else {
                    td.textContent = String(value);
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    // ---------------- UTIL ----------------
    function animateValue(obj, start, end, duration) {
        let startTime = null;

        function step(t) {
            if (!startTime) startTime = t;
            const progress = Math.min((t - startTime) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) requestAnimationFrame(step);
        }

        requestAnimationFrame(step);
    }

    function formatLabel(str) {
        return (str || "")
            .split('_')
            .map(w => w.charAt(0).toUpperCase() + w.slice(1))
            .join(' ');
    }

});