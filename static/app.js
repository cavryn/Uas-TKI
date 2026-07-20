
class SentimentSearchApp {
    constructor() {
        this.isReady = false;
        this.init();
    }

    init() {
        this.statusBar = document.getElementById('statusBar');
        this.statusText = document.getElementById('statusText');
        this.statsDisplay = document.getElementById('statsDisplay');
        this.docCount = document.getElementById('docCount');
        this.sentimentDist = document.getElementById('sentimentDist');
        this.searchInput = document.getElementById('searchInput');
        this.searchButton = document.getElementById('searchButton');
        this.topKSelect = document.getElementById('topK');
        this.exampleContainer = document.getElementById('exampleContainer');
        this.resultsSection = document.getElementById('resultsSection');
        this.sentimentCard = document.getElementById('sentimentCard');
        this.documentsContainer = document.getElementById('documentsContainer');
        this.loadingOverlay = document.getElementById('loadingOverlay');

        this.sentimentBadge = document.getElementById('sentimentBadge');
        this.queryText = document.getElementById('queryText');
        this.confidenceText = document.getElementById('confidenceText');
        this.votePositive = document.getElementById('votePositive');
        this.voteNegative = document.getElementById('voteNegative');
        this.voteNeutral = document.getElementById('voteNeutral');
        this.explanationText = document.getElementById('explanationText');

        this.searchButton.addEventListener('click', () => this.performClassification());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performClassification();
        });

        this.checkStatus();
        this.loadExamples();
    }

    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            if (data.ready) {
                this.isReady = true;
                this.statusBar.classList.remove('loading');
                this.statusBar.classList.add('ready');
                this.statusText.textContent = 'System ready';
                
                this.searchInput.disabled = false;
                this.searchButton.disabled = false;
                this.searchInput.focus();

                if (data.stats) {
                    this.statsDisplay.style.display = 'flex';
                    this.docCount.textContent = `Documents: ${data.stats.total_documents}`;
                    const dist = data.stats.sentiment_distribution;
                    this.sentimentDist.textContent = `Sentiments: ${dist.baik} Baik, ${dist.buruk} Buruk, ${dist.netral} Netral`;
                }
            } else {
                setTimeout(() => this.checkStatus(), 2000);
            }
        } catch (error) {
            console.error('Status check error:', error);
            this.statusBar.classList.remove('loading');
            this.statusBar.classList.add('error');
            this.statusText.textContent = 'Error initializing system';
        }
    }

    async loadExamples() {
        try {
            const response = await fetch('/api/example-queries');
            const data = await response.json();

            if (data.examples) {
                this.exampleContainer.innerHTML = '';
                data.examples.forEach(query => {
                    const btn = document.createElement('button');
                    btn.className = 'example-query';
                    btn.textContent = query;
                    btn.addEventListener('click', () => {
                        this.searchInput.value = query;
                        if (this.isReady) this.performClassification();
                    });
                    this.exampleContainer.appendChild(btn);
                });
            }
        } catch (error) {
            console.error('Error loading examples:', error);
        }
    }

    async performClassification() {
        const query = this.searchInput.value.trim();
        if (!query || !this.isReady) return;

        this.loadingOverlay.style.display = 'flex';
        this.resultsSection.style.display = 'none';

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(this.topKSelect.value)
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayResults(data);
            } else {
                this.showError(data.error || 'Classification failed');
            }
        } catch (error) {
            console.error('Classification error:', error);
            this.showError('Network error occurred');
        } finally {
            this.loadingOverlay.style.display = 'none';
        }
    }

    displayResults(data) {
        this.queryText.textContent = data.query;
        this.confidenceText.textContent = (data.confidence * 100).toFixed(1) + '%';

        this.sentimentBadge.textContent = data.sentiment_label;
        this.sentimentBadge.className = 'sentiment-badge';
        if (data.sentiment === 'baik') {
            this.sentimentBadge.classList.add('sentiment-positive');
        } else if (data.sentiment === 'buruk') {
            this.sentimentBadge.classList.add('sentiment-negative');
        } else {
            this.sentimentBadge.classList.add('sentiment-neutral');
        }

        this.votePositive.textContent = data.votes.baik;
        this.voteNegative.textContent = data.votes.buruk;
        this.voteNeutral.textContent = data.votes.netral;

        this.explanationText.textContent = data.explanation;

        this.documentsContainer.innerHTML = '';
        data.all_documents.forEach((doc, index) => {
            const card = this.createDocumentCard(doc, index + 1);
            this.documentsContainer.appendChild(card);
        });

        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    createDocumentCard(doc, rank) {
        const card = document.createElement('div');
        card.className = 'result-card';

        const header = document.createElement('div');
        header.className = 'result-header';

        const title = document.createElement('div');
        title.className = 'result-title';
        title.innerHTML = `
            <span class="rank-badge">${rank}</span>
            <span class="doc-id">Document D${doc.doc_id}</span>
        `;

        const score = document.createElement('div');
        score.className = 'result-score';
        score.textContent = doc.similarity.toFixed(4);

        header.appendChild(title);
        header.appendChild(score);

        const scoreDetails = document.createElement('div');
        scoreDetails.className = 'score-details';
        
        const sentimentClass = doc.sentiment === 'baik' ? 'sentiment-positive' : 
                              doc.sentiment === 'buruk' ? 'sentiment-negative' : 
                              'sentiment-neutral';
        const sentimentLabel = doc.sentiment === 'baik' ? 'BAIK (Positif)' :
                              doc.sentiment === 'buruk' ? 'BURUK (Negatif)' :
                              'NETRAL';
        
        scoreDetails.innerHTML = `
            <div class="score-item">
                <span class="score-label">Similarity</span>
                <span class="score-value">${doc.similarity.toFixed(4)}</span>
            </div>
            <div class="score-item">
                <span class="score-label">Sentiment</span>
                <span class="score-value ${sentimentClass}">${sentimentLabel}</span>
            </div>
            <div class="score-item">
                <span class="score-label">Sentiment Confidence</span>
                <span class="score-value">${doc.sentiment_confidence.toFixed(4)}</span>
            </div>
        `;

        const content = document.createElement('div');
        content.className = 'result-content';
        const preview = doc.document.length > 200 ? doc.document.substring(0, 200) + '...' : doc.document;
        content.textContent = preview;

        card.appendChild(header);
        card.appendChild(scoreDetails);
        card.appendChild(content);

        return card;
    }

    showError(message) {
        this.documentsContainer.innerHTML = `
            <div class="result-card" style="border-color: var(--error);">
                <p style="color: var(--error); font-weight: 500;">${message}</p>
            </div>
        `;
        this.resultsSection.style.display = 'block';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SentimentSearchApp();
});
