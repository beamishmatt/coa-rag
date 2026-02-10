class InvestigativeAI {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.currentConversation = [];
        this.isProcessing = false;
        this.currentStreamingMessage = null;
        this.streamingContent = '';
        
        this.initializeElements();
        this.initializeEventListeners();
        this.connectWebSocket();
        this.loadDocuments();
        this.updateSendButtonState();
    }

    initializeElements() {
        this.elements = {
            messageInput: document.getElementById('messageInput'),
            sendButton: document.getElementById('sendButton'),
            messagesContainer: document.getElementById('messagesContainer'),
            welcomeScreen: document.getElementById('welcomeScreen'),
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            documentList: document.getElementById('documentList'),
            progressModal: document.getElementById('progressModal'),
            progressFill: document.getElementById('progressFill'),
            progressStatus: document.getElementById('progressStatus'),
            progressTitle: document.getElementById('progressTitle')
        };
    }

    initializeEventListeners() {
        // Message input
        this.elements.messageInput.addEventListener('input', () => {
            this.autoResize();
            this.updateSendButtonState();
        });

        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // File upload
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });

        this.elements.fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Drag and drop
        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });

        this.elements.uploadArea.addEventListener('dragleave', () => {
            this.elements.uploadArea.classList.remove('dragover');
        });

        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateSendButtonState();
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateSendButtonState();
            
            // Reconnect after 3 seconds
            setTimeout(() => {
                this.connectWebSocket();
            }, 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'stage':
                this.updateLoadingState(data.stage, data.content);
                break;
            
            case 'worker_progress':
                this.updateWorkerProgress(data.worker, data.total, data.status);
                break;
            
            case 'stream_start':
                this.startStreamingResponse();
                break;
            
            case 'chunk':
                this.appendStreamChunk(data.content);
                break;
            
            case 'stream_end':
                this.finishStreamingResponse(data.graph_data);
                break;
            
            case 'response':
                // Fallback for non-streaming responses
                this.removeLoadingIndicator();
                this.addMessage('ai', data.content);
                this.finishProcessing();
                break;
            
            case 'error':
                this.removeLoadingIndicator();
                this.addMessage('ai', `❌ ${data.content}`, true);
                this.finishProcessing();
                break;
        }
    }

    async sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message || this.isProcessing || !this.isConnected) return;

        // Hide welcome screen
        this.hideWelcomeScreen();

        // Add user message
        this.addMessage('user', message);
        
        // Clear input
        this.elements.messageInput.value = '';
        this.autoResize();

        // Update state
        this.isProcessing = true;
        this.updateSendButtonState();

        // Show loading indicator
        this.showLoadingIndicator();

        // Send to WebSocket with conversation history
        this.ws.send(JSON.stringify({
            type: 'question',
            content: message,
            history: this.getConversationHistory()
        }));
    }

    getConversationHistory() {
        // Return last N turns of conversation for context (limit to avoid token overflow)
        const maxTurns = 10;
        const history = [];
        
        for (let i = 0; i < this.currentConversation.length && history.length < maxTurns * 2; i++) {
            const msg = this.currentConversation[i];
            history.push({
                role: msg.sender === 'user' ? 'user' : 'assistant',
                content: msg.content
            });
        }
        
        return history;
    }

    hideWelcomeScreen() {
        if (this.elements.welcomeScreen) {
            this.elements.welcomeScreen.style.display = 'none';
        }
    }

    addMessage(sender, content, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        if (isError) messageDiv.classList.add('error-message');

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (sender === 'ai') {
            contentDiv.innerHTML = this.formatMessage(content);
        } else {
            contentDiv.textContent = content;
        }

        messageDiv.appendChild(contentDiv);

        this.elements.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        // Store in conversation
        this.currentConversation.push({
            sender,
            content,
            timestamp: new Date()
        });

        return messageDiv;
    }

    showLoadingIndicator(mode = 'vector') {
        // Remove any existing loading indicator
        this.removeLoadingIndicator();

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'typing-indicator';
        loadingDiv.id = 'loadingIndicator';
        loadingDiv.dataset.mode = mode;

        if (mode === 'graph') {
            // Simpler loading for graph/knowledge base queries
            loadingDiv.innerHTML = `
                <div class="progress-steps" id="progressSteps">
                    <div class="progress-step active" data-step="graph">
                        <i class="fas fa-spinner"></i>
                        <span>Querying knowledge graph...</span>
                    </div>
                    <div class="progress-step pending" data-step="streaming">
                        <i class="fas fa-circle"></i>
                        <span>Generating response</span>
                    </div>
                </div>
            `;
        } else {
            // Full loading for vector/CoA queries
            loadingDiv.innerHTML = `
                <div class="progress-steps" id="progressSteps">
                    <div class="progress-step active" data-step="workers">
                        <i class="fas fa-spinner"></i>
                        <span>Analyzing documents...</span>
                    </div>
                    <div class="progress-step pending" data-step="synthesizing">
                        <i class="fas fa-circle"></i>
                        <span>Synthesizing findings</span>
                    </div>
                    <div class="progress-step pending" data-step="streaming">
                        <i class="fas fa-circle"></i>
                        <span>Generating response</span>
                    </div>
                </div>
            `;
        }

        this.elements.messagesContainer.appendChild(loadingDiv);
        this.scrollToBottom();
    }

    updateLoadingState(stage, message) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        const mode = loadingIndicator?.dataset.mode || 'vector';
        
        // Handle graph mode - switch to graph loading if receiving graph stage
        if (stage === 'graph' && mode !== 'graph') {
            this.showLoadingIndicator('graph');
            return;
        }
        
        const steps = document.querySelectorAll('.progress-step');
        
        steps.forEach(step => {
            const stepStage = step.dataset.step;
            const icon = step.querySelector('i');
            const span = step.querySelector('span');
            
            if (stepStage === stage) {
                step.className = 'progress-step active';
                icon.className = 'fas fa-spinner';
                // Update message if provided
                if (message && span) {
                    span.textContent = message;
                }
            } else if (this.getStageOrder(stepStage, mode) < this.getStageOrder(stage, mode)) {
                step.className = 'progress-step completed';
                icon.className = 'fas fa-check';
            } else {
                step.className = 'progress-step pending';
                icon.className = 'fas fa-circle';
            }
        });
    }

    getStageOrder(stage, mode = 'vector') {
        if (mode === 'graph') {
            const order = { 'graph': 0, 'streaming': 1 };
            return order[stage] ?? -1;
        }
        const order = { 'workers': 0, 'synthesizing': 1, 'streaming': 2 };
        return order[stage] ?? -1;
    }

    updateWorkerProgress(worker, total, status) {
        const workerStep = document.querySelector('[data-step="workers"] span');
        if (workerStep) {
            workerStep.textContent = `Worker ${worker}/${total} analyzing...`;
        }
    }

    startStreamingResponse() {
        // Remove loading indicator
        this.removeLoadingIndicator();
        
        // Create streaming message container
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';
        messageDiv.id = 'streamingMessage';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.id = 'streamingContent';
        contentDiv.innerHTML = '<span class="streaming-cursor"></span>';

        messageDiv.appendChild(contentDiv);

        this.elements.messagesContainer.appendChild(messageDiv);
        this.currentStreamingMessage = messageDiv;
        this.streamingContent = '';
        this.scrollToBottom();
    }

    appendStreamChunk(chunk) {
        if (!chunk) return;
        
        this.streamingContent += chunk;
        const contentDiv = document.getElementById('streamingContent');
        if (contentDiv) {
            // Format and render with cursor
            contentDiv.innerHTML = this.formatMessage(this.streamingContent) + '<span class="streaming-cursor"></span>';
            this.scrollToBottom();
        }
    }

    finishStreamingResponse(graphData = null) {
        const contentDiv = document.getElementById('streamingContent');
        const messageDiv = document.getElementById('streamingMessage');
        
        if (contentDiv) {
            // Remove cursor and finalize formatting
            contentDiv.innerHTML = this.formatMessage(this.streamingContent);
            // Remove the ID so it doesn't interfere with future streaming responses
            contentDiv.removeAttribute('id');
        }
        
        // If graph data is present, add toggle and graph container
        if (graphData && graphData.nodes && graphData.nodes.length > 0 && messageDiv) {
            this.addGraphVisualization(messageDiv, graphData);
        }
        
        if (messageDiv) {
            // Remove the ID so it doesn't interfere with future streaming responses
            messageDiv.removeAttribute('id');
        }

        // Store in conversation
        if (this.streamingContent) {
            this.currentConversation.push({
                sender: 'ai',
                content: this.streamingContent,
                timestamp: new Date(),
                graphData: graphData // Store graph data with conversation
            });
        }

        this.currentStreamingMessage = null;
        this.streamingContent = '';
        this.finishProcessing();
    }

    /**
     * Add graph visualization toggle and container to a message
     */
    addGraphVisualization(messageDiv, graphData) {
        const contentDiv = messageDiv.querySelector('.message-content');
        if (!contentDiv) return;

        // Generate unique ID for this graph
        const graphId = 'graph-' + Date.now();

        // Create toggle container
        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'view-toggle-container';
        toggleContainer.innerHTML = `
            <div class="view-toggle">
                <button class="view-toggle-btn active" data-view="text" onclick="app.toggleGraphView('${graphId}', 'text')">
                    <i class="fas fa-align-left"></i> Text
                </button>
                <button class="view-toggle-btn" data-view="graph" onclick="app.toggleGraphView('${graphId}', 'graph')">
                    <i class="fas fa-project-diagram"></i> Graph
                </button>
            </div>
        `;

        // Wrap existing content in text-view container
        const textContent = contentDiv.innerHTML;
        contentDiv.innerHTML = '';
        
        const textView = document.createElement('div');
        textView.className = 'text-view active';
        textView.id = `${graphId}-text`;
        textView.innerHTML = textContent;

        // Create graph container
        const graphView = document.createElement('div');
        graphView.className = 'graph-view';
        graphView.id = `${graphId}-graph`;
        graphView.innerHTML = `
            <div class="graph-container" id="${graphId}-container">
                <div class="graph-loading">
                    <i class="fas fa-spinner fa-spin"></i> Loading graph...
                </div>
            </div>
            <div class="graph-detail-panel" id="${graphId}-detail"></div>
        `;

        // Insert toggle before content
        contentDiv.appendChild(toggleContainer);
        contentDiv.appendChild(textView);
        contentDiv.appendChild(graphView);

        // Store graph data for later rendering
        if (!window.graphDataStore) {
            window.graphDataStore = {};
        }
        window.graphDataStore[graphId] = graphData;

        console.log(`Graph data stored for ${graphId}:`, graphData);
    }

    /**
     * Toggle between text and graph views
     */
    toggleGraphView(graphId, view) {
        const textView = document.getElementById(`${graphId}-text`);
        const graphView = document.getElementById(`${graphId}-graph`);
        const toggleBtns = document.querySelectorAll(`[onclick*="${graphId}"]`);

        if (!textView || !graphView) return;

        // Update button states
        toggleBtns.forEach(btn => {
            if (btn.dataset.view === view) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        if (view === 'text') {
            textView.classList.add('active');
            graphView.classList.remove('active');
        } else {
            textView.classList.remove('active');
            graphView.classList.add('active');

            // Render graph if not already rendered
            const container = document.getElementById(`${graphId}-container`);
            if (container && !container.dataset.rendered) {
                const graphData = window.graphDataStore[graphId];
                if (graphData && window.RelationshipGraph) {
                    container.innerHTML = ''; // Clear loading message
                    const graph = new window.RelationshipGraph(container, graphData, `${graphId}-detail`);
                    graph.render();
                    container.dataset.rendered = 'true';
                }
            }
        }

        this.scrollToBottom();
    }

    removeLoadingIndicator() {
        const indicator = document.getElementById('loadingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    finishProcessing() {
        this.isProcessing = false;
        this.updateSendButtonState();
        this.scrollToBottom();
        
        // Focus input for next message
        this.elements.messageInput.focus();
    }

    /**
     * Extract follow-up suggestions from the response content.
     * Looks for the "Suggested Follow-ups:" section and extracts the suggestions.
     * Handles multiple formats:
     *   - With or without --- separator
     *   - With or without ** bold markers
     *   - With or without - bullet markers
     */
    extractFollowUpSuggestions(content) {
        if (!content) return { mainContent: content, suggestions: [] };
        
        // Try multiple patterns to match different formats
        const patterns = [
            // Format 1: ---\n**Suggested Follow-ups:**\n- suggestion (ideal format)
            /\n---\s*\n\*\*Suggested Follow-ups:\*\*\s*\n((?:[-*] .+\n?)+)/i,
            // Format 2: **Suggested Follow-ups:**\n- suggestion (no separator)
            /\n\*\*Suggested Follow-ups:\*\*\s*\n((?:[-*] .+\n?)+)/i,
            // Format 3: Suggested Follow-ups:\n- suggestion (no bold)
            /\nSuggested Follow-ups:\s*\n((?:[-*] .+\n?)+)/i,
            // Format 4: ---\n**Suggested Follow-ups:**\nsuggestion (no bullets)
            /\n---\s*\n\*\*Suggested Follow-ups:\*\*\s*\n((?:[^\n-*][^\n]+\n?)+)/i,
            // Format 5: Suggested Follow-ups:\nsuggestion (plain text, no bullets)
            /\nSuggested Follow-ups:\s*\n((?:(?!^#|^---|^\*\*)[^\n]+\n?)+)/im,
        ];
        
        let match = null;
        let matchedPattern = null;
        
        for (const pattern of patterns) {
            match = content.match(pattern);
            if (match) {
                matchedPattern = pattern;
                break;
            }
        }
        
        if (!match) {
            return { mainContent: content, suggestions: [] };
        }
        
        // Extract the suggestions list
        const suggestionsBlock = match[1];
        const suggestions = suggestionsBlock
            .split('\n')
            .map(line => line.replace(/^[-*]\s*/, '').trim())  // Remove bullet markers
            .filter(line => line.length > 0 && !line.match(/^#|^---|^\*\*/));  // Filter empty/header lines
        
        // Remove the suggestions section from main content
        const mainContent = content.replace(matchedPattern, '');
        
        return { mainContent, suggestions };
    }

    /**
     * Render follow-up suggestions as clickable chips.
     */
    renderSuggestionChips(suggestions) {
        if (!suggestions || suggestions.length === 0) return '';
        
        const chips = suggestions.map(suggestion => {
            // Escape the suggestion for use in onclick attribute
            const escapedSuggestion = suggestion
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
            return `<button class="suggestion-chip" onclick="app.useSuggestion('${escapedSuggestion}')" title="Click to ask this question">${suggestion}</button>`;
        }).join('');
        
        return `
            <div class="suggestions-container">
                <div class="suggestions-label">Suggested Follow-ups:</div>
                <div class="suggestions-chips">${chips}</div>
            </div>
        `;
    }

    /**
     * Handle clicking on a suggestion chip - populate the input and focus it.
     */
    useSuggestion(suggestion) {
        // Decode HTML entities back to normal characters
        const decoded = suggestion
            .replace(/&quot;/g, '"')
            .replace(/&#39;/g, "'")
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>');
        
        this.elements.messageInput.value = decoded;
        this.autoResize();
        this.updateSendButtonState();
        this.elements.messageInput.focus();
        
        // Scroll to the input area
        this.elements.messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    formatMessage(content) {
        if (!content) return '';
        
        // Extract follow-up suggestions before processing
        const { mainContent, suggestions } = this.extractFollowUpSuggestions(content);
        
        // Escape HTML first
        let html = mainContent
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // Handle code blocks first (before other processing)
        html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });
        
        // Split into lines for block-level processing
        const lines = html.split('\n');
        const processedLines = [];
        let inList = false;
        let listType = null;
        
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];
            
            // Headers (check longer prefixes first)
            if (line.match(/^#### /)) {
                if (inList) { processedLines.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
                processedLines.push(`<h4>${line.slice(5)}</h4>`);
                continue;
            }
            if (line.match(/^### /)) {
                if (inList) { processedLines.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
                processedLines.push(`<h3>${line.slice(4)}</h3>`);
                continue;
            }
            if (line.match(/^## /)) {
                if (inList) { processedLines.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
                processedLines.push(`<h2>${line.slice(3)}</h2>`);
                continue;
            }
            if (line.match(/^# /)) {
                if (inList) { processedLines.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
                processedLines.push(`<h1>${line.slice(2)}</h1>`);
                continue;
            }
            
            // Horizontal rule
            if (line.match(/^---+$/)) {
                if (inList) { processedLines.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
                processedLines.push('<hr>');
                continue;
            }
            
            // Unordered list
            if (line.match(/^[\-\*] /)) {
                if (!inList || listType !== 'ul') {
                    if (inList) processedLines.push('</ol>');
                    processedLines.push('<ul>');
                    inList = true;
                    listType = 'ul';
                }
                processedLines.push(`<li>${line.slice(2)}</li>`);
                continue;
            }
            
            // Ordered list
            if (line.match(/^\d+\. /)) {
                if (!inList || listType !== 'ol') {
                    if (inList) processedLines.push('</ul>');
                    processedLines.push('<ol>');
                    inList = true;
                    listType = 'ol';
                }
                processedLines.push(`<li>${line.replace(/^\d+\. /, '')}</li>`);
                continue;
            }
            
            // Blockquote
            if (line.match(/^&gt; /)) {
                if (inList) { processedLines.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
                processedLines.push(`<blockquote>${line.slice(5)}</blockquote>`);
                continue;
            }
            
            // Close list if we hit a non-list line
            if (inList && line.trim() !== '') {
                processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');
                inList = false;
            }
            
            // Regular paragraph or empty line
            if (line.trim() === '') {
                processedLines.push('<br>');
            } else {
                processedLines.push(`<p>${line}</p>`);
            }
        }
        
        // Close any open list
        if (inList) {
            processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');
        }
        
        html = processedLines.join('');
        
        // Inline formatting
        html = html
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/__(.+?)__/g, '<strong>$1</strong>')
            .replace(/_(.+?)_/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Clean up empty paragraphs and extra breaks
        html = html
            .replace(/<p><\/p>/g, '')
            .replace(/<br><br><br>/g, '<br>')
            .replace(/<br><br>/g, '<br>')
            .replace(/<\/h(\d)><br>/g, '</h$1>')
            .replace(/<\/ul><br>/g, '</ul>')
            .replace(/<\/ol><br>/g, '</ol>')
            .replace(/<\/blockquote><br>/g, '</blockquote>')
            .replace(/<hr><br>/g, '<hr>')
            .replace(/<br><h/g, '<h')
            .replace(/<br><hr/g, '<hr');
        
        // Append suggestion chips if there are any
        if (suggestions && suggestions.length > 0) {
            html += this.renderSuggestionChips(suggestions);
        }
        
        return html;
    }

    scrollToBottom() {
        requestAnimationFrame(() => {
            this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
        });
    }

    async handleFileUpload(files) {
        const formData = new FormData();
        
        for (let file of files) {
            if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
                formData.append('files', file);
            }
        }

        if (formData.getAll('files').length === 0) {
            alert('Please select PDF files only.');
            return;
        }

        try {
            this.showUploadProgress('Uploading documents...');
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const errorMsg = errorData.detail || `Upload failed: ${response.statusText}`;
                throw new Error(errorMsg);
            }

            const result = await response.json();
            this.hideUploadProgress();
            
            await this.loadDocuments();
            
            // Show success in chat
            this.hideWelcomeScreen();
            this.addMessage('ai', `✅ Successfully uploaded ${result.uploaded_files} document(s). You can now ask questions about them.`);
            
        } catch (error) {
            this.hideUploadProgress();
            let errorMessage = error.message;
            
            if (error.message.includes('OpenAI API key')) {
                errorMessage = '🔑 Please add your OpenAI API key to the .env file and restart the server.';
            }
            
            this.hideWelcomeScreen();
            this.addMessage('ai', `❌ ${errorMessage}`, true);
        }
    }

    showUploadProgress(message) {
        this.elements.progressTitle.textContent = 'Uploading Documents';
        this.elements.progressStatus.textContent = message;
        this.elements.progressFill.style.width = '0%';
        this.elements.progressModal.classList.add('show');
        
        // Animate progress
        let progress = 0;
        this.uploadProgressInterval = setInterval(() => {
            progress = Math.min(progress + Math.random() * 15, 90);
            this.elements.progressFill.style.width = `${progress}%`;
        }, 200);
    }

    hideUploadProgress() {
        clearInterval(this.uploadProgressInterval);
        this.elements.progressFill.style.width = '100%';
        setTimeout(() => {
            this.elements.progressModal.classList.remove('show');
        }, 300);
    }

    async loadDocuments() {
        try {
            const response = await fetch('/api/documents');
            const data = await response.json();
            this.renderDocumentList(data.documents);
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    renderDocumentList(documents) {
        this.elements.documentList.innerHTML = '';
        
        if (documents.length === 0) {
            return;
        }
        
        documents.forEach(doc => {
            const docDiv = document.createElement('div');
            docDiv.className = 'document-item';
            
            docDiv.innerHTML = `
                <div class="document-info">
                    <div class="document-name" title="${doc.filename}">${doc.filename}</div>
                    <div class="document-status">${doc.status}</div>
                </div>
                <div class="document-actions">
                    <button onclick="app.deleteDocument('${doc.id}')" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            
            this.elements.documentList.appendChild(docDiv);
        });
    }

    async deleteDocument(fileId) {
        if (!confirm('Delete this document?')) {
            return;
        }

        try {
            const response = await fetch(`/api/documents/${fileId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Delete failed');
            }

            await this.loadDocuments();
            
        } catch (error) {
            alert(`Delete failed: ${error.message}`);
        }
    }

    updateSendButtonState() {
        const hasText = this.elements.messageInput.value.trim().length > 0;
        this.elements.sendButton.disabled = !hasText || this.isProcessing || !this.isConnected;
    }

    autoResize() {
        const input = this.elements.messageInput;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 150) + 'px';
    }
}

// Global functions for HTML onclick handlers
function sendMessage() {
    app.sendMessage();
}

function startNewChat() {
    app.currentConversation = [];
    
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    // Remove all messages
    const messages = messagesContainer.querySelectorAll('.message, .typing-indicator');
    messages.forEach(msg => msg.remove());
    
    // Show welcome screen
    if (welcomeScreen) {
        welcomeScreen.style.display = 'flex';
    }
    
    // Clear and focus input
    document.getElementById('messageInput').value = '';
    app.autoResize();
    app.updateSendButtonState();
    document.getElementById('messageInput').focus();
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    const toggleBtn = document.getElementById('sidebarToggle');
    
    if (sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
        backdrop.classList.remove('show');
        setTimeout(() => {
            toggleBtn.classList.remove('hidden');
        }, 150);
    } else {
        sidebar.classList.add('open');
        backdrop.classList.add('show');
        toggleBtn.classList.add('hidden');
    }
}

function toggleReadme() {
    const readmeDrawer = document.getElementById('readmeDrawer');
    const backdrop = document.getElementById('readmeBackdrop');
    const toggleBtn = document.getElementById('readmeToggle');
    
    if (readmeDrawer.classList.contains('open')) {
        readmeDrawer.classList.remove('open');
        backdrop.classList.remove('show');
        setTimeout(() => {
            toggleBtn.classList.remove('hidden');
        }, 150);
    } else {
        readmeDrawer.classList.add('open');
        backdrop.classList.add('show');
        toggleBtn.classList.add('hidden');
    }
}

function exportConversation() {
    if (app.currentConversation.length === 0) {
        alert('No conversation to export.');
        return;
    }

    let exportText = `# Investigation Report\n\nExported: ${new Date().toLocaleString()}\n\n`;
    
    app.currentConversation.forEach((msg, index) => {
        if (msg.sender === 'user') {
            exportText += `## Question ${Math.floor(index/2) + 1}\n${msg.content}\n\n`;
        } else {
            exportText += `## Analysis\n${msg.content}\n\n---\n\n`;
        }
    });

    const blob = new Blob([exportText], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `investigation-report-${new Date().toISOString().split('T')[0]}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

// Initialize
const app = new InvestigativeAI();
window.app = app;
