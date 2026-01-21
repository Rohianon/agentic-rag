// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const browseBtn = document.getElementById('browse-btn');
const uploadStatus = document.getElementById('upload-status');
const documentsList = document.getElementById('documents-list');
const refreshDocsBtn = document.getElementById('refresh-docs');
const queryForm = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const queryBtn = document.getElementById('query-btn');
const resultsSection = document.getElementById('results-section');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// API Functions
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
}

async function queryDocuments(query) {
    const response = await fetch('/api/query/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, n_chunks: 5 })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Query failed');
    }

    return response.json();
}

async function listDocuments() {
    const response = await fetch('/api/documents/list');
    return response.json();
}

// UI Functions
function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status ${type}`;
    uploadStatus.classList.remove('hidden');
}

function hideStatus() {
    uploadStatus.classList.add('hidden');
}

async function refreshDocumentsList() {
    try {
        const data = await listDocuments();

        if (data.documents.length === 0) {
            documentsList.innerHTML = '<p class="empty-state">No documents indexed yet</p>';
            return;
        }

        documentsList.innerHTML = data.documents.map(doc => `
            <div class="document-item">
                <div class="document-info">
                    <span class="document-icon">📄</span>
                    <div>
                        <div>${doc.filename}</div>
                        <div class="document-meta">${doc.total_pages} pages, ${doc.pages_with_tables} with tables</div>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        documentsList.innerHTML = '<p class="empty-state">Failed to load documents</p>';
    }
}

function renderResults(data) {
    // Confidence
    const confidence = Math.round(data.confidence_score * 100);
    document.getElementById('confidence-fill').style.width = `${confidence}%`;
    document.getElementById('confidence-value').textContent = `${confidence}%`;

    // Summary
    document.getElementById('summary').textContent = data.summary;

    // Key Findings
    const findingsList = document.getElementById('findings');
    findingsList.innerHTML = data.key_findings.map(f => `<li>${f}</li>`).join('');

    // Extracted Data
    const extractedSection = document.getElementById('extracted-section');
    const extractedData = document.getElementById('extracted-data');

    if (Object.keys(data.extracted_data).length > 0) {
        extractedData.innerHTML = Object.entries(data.extracted_data).map(([key, val]) => `
            <div class="data-card">
                <div class="label">${key}</div>
                <div class="value">${val.value || val}${val.unit ? ` ${val.unit}` : ''}</div>
            </div>
        `).join('');
        extractedSection.classList.remove('hidden');
    } else {
        extractedSection.classList.add('hidden');
    }

    // Risk Flags
    const risksSection = document.getElementById('risks-section');
    const riskFlags = document.getElementById('risk-flags');

    if (data.risk_flags.length > 0) {
        riskFlags.innerHTML = data.risk_flags.map(rf => `
            <div class="risk-flag ${rf.severity.toLowerCase()}">
                <div class="severity">${rf.severity}</div>
                <div>${rf.message}</div>
            </div>
        `).join('');
        risksSection.classList.remove('hidden');
    } else {
        risksSection.classList.add('hidden');
    }

    // Citations
    const citations = document.getElementById('citations');
    citations.innerHTML = data.citations.map(c => `
        <div class="citation">
            <div class="citation-header">
                <span class="citation-source">[${c.id}] ${c.source_file}, Page ${c.page}</span>
                <span class="citation-relevance">${Math.round(c.relevance * 100)}% relevant</span>
            </div>
            <div class="citation-excerpt">"${c.excerpt}"</div>
        </div>
    `).join('');

    // Show results
    loading.classList.add('hidden');
    results.classList.remove('hidden');
}

// Event Handlers
browseBtn.addEventListener('click', (e) => {
    e.preventDefault();
    fileInput.click();
});

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', async (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        await handleFileUpload(file);
    } else {
        showStatus('Please upload a PDF file', 'error');
    }
});

fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (file) {
        await handleFileUpload(file);
    }
});

async function handleFileUpload(file) {
    showStatus(`Uploading ${file.name}...`, 'loading');

    try {
        const result = await uploadFile(file);
        showStatus(`${result.message}`, 'success');
        await refreshDocumentsList();
    } catch (error) {
        showStatus(error.message, 'error');
    }

    fileInput.value = '';
}

refreshDocsBtn.addEventListener('click', refreshDocumentsList);

queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const query = queryInput.value.trim();
    if (!query) return;

    resultsSection.classList.remove('hidden');
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    queryBtn.disabled = true;

    try {
        const data = await queryDocuments(query);
        renderResults(data);
    } catch (error) {
        loading.classList.add('hidden');
        results.innerHTML = `<p class="error">${error.message}</p>`;
        results.classList.remove('hidden');
    } finally {
        queryBtn.disabled = false;
    }
});

// Initialize
refreshDocumentsList();
