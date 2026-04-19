import { useState, useRef, useEffect } from 'react'
import { Search, Sparkles, Loader2, Paperclip, SlidersHorizontal, User, Bot, CheckCircle2, Trash2 } from 'lucide-react'
import './App.css'

function App() {
  const [query, setQuery] = useState('')
  const [kValue, setKValue] = useState(3)
  const [chatHistory, setChatHistory] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isClearing, setIsClearing] = useState(false)

  const fileInputRef = useRef(null)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, isLoading])

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    const userMessage = query
    setQuery('')
    setIsLoading(true)
    setChatHistory(prev => [...prev, { type: 'user', text: userMessage }])

    try {
      const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(userMessage)}&k=${kValue}`)
      const data = await response.json()
      setChatHistory(prev => [...prev, { type: 'ai', results: data.results }])
    } catch (error) {
      setChatHistory(prev => [...prev, { type: 'error', text: "Failed to connect to the backend." }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (e) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploading(true)
    const formData = new FormData()
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i])
    }

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setChatHistory(prev => [...prev, { type: 'system', text: data.message || data.error }])
    } catch (error) {
      setChatHistory(prev => [...prev, { type: 'error', text: "Failed to upload files." }])
    } finally {
      setIsUploading(false)
      e.target.value = null
    }
  }

  // ✅ FIX 3 / NEW: Clear database handler
  const handleClearDatabase = async () => {
    if (!window.confirm("Are you sure you want to clear the entire database? This cannot be undone.")) return

    setIsClearing(true)
    try {
      const response = await fetch('http://localhost:8000/clear', { method: 'DELETE' })
      const data = await response.json()
      setChatHistory(prev => [...prev, { type: 'system', text: data.message || data.error }])
    } catch (error) {
      setChatHistory(prev => [...prev, { type: 'error', text: "Failed to clear database." }])
    } finally {
      setIsClearing(false)
    }
  }

  // ✅ FIX 3: Helper — maps a cross-encoder score to a colour for the badge
  const getScoreColor = (score) => {
    if (score >= 5) return '#16a34a'   // green  — high confidence
    if (score >= 0) return '#d97706'   // amber  — medium
    return '#dc2626'                   // red    — low confidence
  }

  return (
    <div className="app-layout">
      <header className="top-nav">
        <div className="nav-brand">
          <Sparkles className="logo-icon" size={20} />
          <h2>Semantic Search MVP</h2>
        </div>

        <div className="nav-controls">
          <div className="settings-panel">
            <SlidersHorizontal size={16} color="#64748b" />
            <label>Top Results (k = {kValue})</label>
            <input
              type="range"
              min="1"
              max="10"
              value={kValue}
              onChange={(e) => setKValue(e.target.value)}
              className="k-slider"
            />
          </div>

          {/* ✅ NEW: Clear Database button */}
          <button
            className="clear-btn"
            onClick={handleClearDatabase}
            disabled={isClearing || isLoading || isUploading}
            title="Clear entire vector database"
          >
            {isClearing ? <Loader2 className="spinner" size={16} /> : <Trash2 size={16} />}
            {isClearing ? 'Clearing…' : 'Clear DB'}
          </button>
        </div>
      </header>

      <main className="chat-container">
        {chatHistory.length === 0 ? (
          <div className="welcome-screen">
            <Sparkles size={48} color="#3b82f6" style={{ marginBottom: '1rem' }} />
            <h1>How can I help you search?</h1>
            <p>Upload your text documents and ask questions based on their semantic meaning.</p>
          </div>
        ) : (
          <div className="chat-feed">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`message-wrapper ${msg.type}`}>
                <div className="message-avatar">
                  {msg.type === 'user' && <User size={20} />}
                  {msg.type === 'ai' && <Bot size={20} color="#3b82f6" />}
                  {msg.type === 'system' && <CheckCircle2 size={20} color="#10b981" />}
                </div>

                <div className="message-content">
                  {(msg.type === 'user' || msg.type === 'system' || msg.type === 'error') && (
                    <p className="plain-text">{msg.text}</p>
                  )}

                  {msg.type === 'ai' && msg.results && msg.results.length === 0 && (
                    <p className="plain-text">No relevant matches found.</p>
                  )}

                  {msg.type === 'ai' && msg.results && msg.results.map((res, i) => (
                    <div key={i} className="result-card">
                      <div className="result-card-header">
                        <span className="result-badge">Match #{i + 1} • {res.source}</span>

                        {/* ✅ FIX 3: Confidence score badge */}
                        <span
                          className="confidence-badge"
                          style={{ color: getScoreColor(res.confidence_score) }}
                          title="Cross-encoder re-ranker confidence score"
                        >
                          Score: {res.confidence_score.toFixed(2)}
                        </span>
                      </div>
                      <p>{res.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="message-wrapper ai">
                <div className="message-avatar"><Bot size={20} color="#3b82f6" /></div>
                <div className="message-content">
                  <Loader2 className="spinner" size={24} color="#3b82f6" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        )}
      </main>

      <div className="bottom-input-area">
        <form onSubmit={handleSearch} className="input-form">
          <input
            type="file"
            multiple
            accept=".txt,.pdf,.docx,.csv,.xlsx,.xls"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />

          <button
            type="button"
            className="action-button upload-btn"
            onClick={() => fileInputRef.current.click()}
            disabled={isUploading || isLoading}
            title="Upload documents (PDF, Word, Excel, CSV, TXT)"
          >
            {isUploading ? <Loader2 className="spinner" size={20} /> : <Paperclip size={20} />}
          </button>

          <input
            type="text"
            className="main-search-input"
            placeholder="Ask a question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading || isUploading}
          />

          <button
            type="submit"
            className="action-button submit-btn"
            disabled={isLoading || isUploading || !query.trim()}
          >
            <Search size={20} />
          </button>
        </form>
        <p className="footer-text">Responses are retrieved directly from your local vector embeddings.</p>
      </div>
    </div>
  )
}

export default App
