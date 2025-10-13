function createParticles() {
            const container = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 20 + 's';
                particle.style.animationDuration = (15 + Math.random() * 10) + 's';
                container.appendChild(particle);
            }
        }
        createParticles();

        // Animate stats on load
        function animateStats() {
            animateValue('accuracy', 0, 94, 2000);
            animateValue('avgTime', 0, 3.2, 2000, 1);
        }
        animateStats();

        function uploadImage() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) return;

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    setTimeout(() => displayResults(data), 1000);
                } else {
                    alert('Error: ' + data.error);
                }
                document.getElementById('loading').style.display = 'none';
            })
            .catch(error => {
                alert('Error: ' + error);
                document.getElementById('loading').style.display = 'none';
            });
        }

        function displayResults(data) {
            document.getElementById('results').style.display = 'block';
            window.scrollTo({ top: document.getElementById('results').offsetTop - 100, behavior: 'smooth' });

            const confidence = data.confidence;

            // Speedometer
            const angle = (confidence / 100) * 180 - 90;
            document.getElementById('needle').style.transform = `translateX(-50%) rotate(${angle}deg)`;
            
            // Gauge progress
            const gaugeProgress = document.getElementById('gaugeProgress');
            const circumference = 251.2;
            const offset = circumference - (confidence / 100) * circumference;
            gaugeProgress.style.strokeDashoffset = offset;

            // Animate confidence
            animateValue('confidenceValue', 0, confidence, 2000, 0, '%');

            // Prediction badge
            document.getElementById('predictionBadge').textContent = data.predicted_class;

            // Images
            document.getElementById('originalImg').src = 'data:image/png;base64,' + data.images.original;
            document.getElementById('enhancedImg').src = 'data:image/png;base64,' + data.images.enhanced;
            document.getElementById('heatmapImg').src = 'data:image/png;base64,' + data.images.heatmap;

            // Circular probabilities
            const colors = ['#5761B2', '#3B9FB5', '#1FC5A8', '#26D0B8'];
            const probGrid = document.getElementById('probabilitiesGrid');
            probGrid.innerHTML = '';
            
            Object.entries(data.all_probabilities).forEach(([cls, prob], index) => {
                const color = colors[index];
                const circle = document.createElement('div');
                circle.className = 'prob-circle';
                circle.innerHTML = `
                    <div class="circle-container">
                        <svg width="200" height="200">
                            <circle class="circle-bg" cx="100" cy="100" r="80"/>
                            <circle class="circle-progress" cx="100" cy="100" r="80"
                                    stroke="${color}"
                                    stroke-dasharray="502.4"
                                    stroke-dashoffset="502.4"
                                    transform="rotate(-90 100 100)"
                                    id="circle-${index}"/>
                        </svg>
                        <div class="circle-text">
                            <div class="circle-percentage" id="percent-${index}">0%</div>
                        </div>
                    </div>
                    <div class="circle-label">${cls}</div>
                `;
                probGrid.appendChild(circle);

                // Animate circle
                setTimeout(() => {
                    const circumference = 502.4;
                    const offset = circumference - (prob / 100) * circumference;
                    document.getElementById(`circle-${index}`).style.strokeDashoffset = offset;
                    animateValue(`percent-${index}`, 0, prob, 1500, 1, '%');
                }, 100 * index);
            });

            // Feature bars
            const featuresSection = document.getElementById('featuresSection');
            featuresSection.innerHTML = '';
            const featureColors = [
                'linear-gradient(90deg, #5761B2, #3B9FB5)',
                'linear-gradient(90deg, #3B9FB5, #1FC5A8)',
                'linear-gradient(90deg, #1FC5A8, #26D0B8)',
                'linear-gradient(90deg, #26D0B8, #1FC5A8)',
                'linear-gradient(90deg, #1FC5A8, #3B9FB5)',
                'linear-gradient(90deg, #3B9FB5, #5761B2)'
            ];
            
            Object.entries(data.glcm_features).forEach(([name, value], index) => {
                const maxVal = 300;
                const percentage = Math.min((value / maxVal) * 100, 100);
                
                const feature = document.createElement('div');
                feature.className = 'feature-item';
                feature.innerHTML = `
                    <div class="feature-header">
                        <span class="feature-name">${name}</span>
                        <span class="feature-value">${value.toFixed(4)}</span>
                    </div>
                    <div class="feature-bar-container">
                        <div class="feature-bar-fill" style="width: ${percentage}%; background: ${featureColors[index]}"></div>
                    </div>
                `;
                featuresSection.appendChild(feature);
            });

            // Medical Insights
            displayMedicalInsights(data);
        }

        function displayMedicalInsights(data) {
            const diseaseInfo = data.disease_info;
            const severityInfo = data.severity_info;
            
            document.getElementById('medicalInsights').style.display = 'block';

            // Disease name and description
            document.getElementById('diseaseName').textContent = diseaseInfo.name;
            document.getElementById('diseaseDesc').textContent = diseaseInfo.description;

            // Urgency badge with dynamic severity
            const urgencyBadge = document.getElementById('urgencyBadge');
            urgencyBadge.textContent = severityInfo.urgency;
            urgencyBadge.style.background = severityInfo.color;
            urgencyBadge.style.color = 'white';

            // Risk indicator with severity info
            const riskIndicator = document.getElementById('riskIndicator');
            const riskScore = severityInfo.risk_score;
            riskIndicator.innerHTML = `
                <div style="margin-bottom: 15px;">Severity: ${severityInfo.level}</div>
                <div style="font-size: 1em; font-weight: 400; margin-bottom: 15px;">${severityInfo.description}</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: white; font-weight: 600;">Risk Score</span>
                    <span style="color: white; font-weight: 700;">${riskScore}/100</span>
                </div>
                <div style="height: 15px; background: rgba(255,255,255,0.2); border-radius: 10px; overflow: hidden;">
                    <div style="height: 100%; width: ${riskScore}%; background: ${severityInfo.color}; transition: width 1.5s;"></div>
                </div>
            `;
            riskIndicator.style.background = severityInfo.color + '33';
            riskIndicator.style.border = `2px solid ${severityInfo.color}`;
            riskIndicator.style.color = 'white';

            // Populate lists
            function populateList(elementId, items) {
                const list = document.getElementById(elementId);
                list.innerHTML = items.map(item => `<li>✓ ${item}</li>`).join('');
            }

            populateList('symptomsList', diseaseInfo.symptoms);
            populateList('causesList', diseaseInfo.causes);
            populateList('treatmentList', diseaseInfo.treatment);
            populateList('lifestyleList', diseaseInfo.lifestyle);

            // Prognosis
            document.getElementById('prognosisText').textContent = diseaseInfo.prognosis;

            // Smooth scroll to insights
            setTimeout(() => {
                document.getElementById('medicalInsights').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 1000);
        }

        function animateValue(id, start, end, duration, decimals = 0, suffix = '') {
            const element = document.getElementById(id);
            const range = end - start;
            const startTime = performance.now();

            function update(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const easeProgress = 1 - Math.pow(1 - progress, 3);
                const value = start + (range * easeProgress);
                element.textContent = value.toFixed(decimals) + suffix;

                if (progress < 1) {
                    requestAnimationFrame(update);
                }
            }

            requestAnimationFrame(update);
        }