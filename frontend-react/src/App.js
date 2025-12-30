import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [mode, setMode] = useState('search'); // 'search' or 'chat'
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Chat state
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [sessionId] = useState(`user_${Date.now()}`);

  const chatEndRef = useRef(null);
  const searchInputRef = useRef(null);

  useEffect(() => {
    if (mode === 'search' && !hasSearched) {
      searchInputRef.current?.focus();
    }
  }, [mode, hasSearched]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setHasSearched(true);

    try {
      const response = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product: searchQuery,
          max_price: null,
          location: null,
          certification: null
        })
      });

      const data = await response.json();
      setSearchResults(data);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMessage = { role: 'user', content: chatInput };
    setMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: chatInput
        })
      });

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.message || 'No response received',
        suppliers: data.suppliers || []
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat failed:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        suppliers: []
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
    setHasSearched(false);
    searchInputRef.current?.focus();
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="logo">SourceUP</h1>
          <div className="mode-toggle">
            <button
              className={mode === 'search' ? 'active' : ''}
              onClick={() => setMode('search')}
            >
              🔍 Search
            </button>
            <button
              className={mode === 'chat' ? 'active' : ''}
              onClick={() => setMode('chat')}
            >
              💬 Chat
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {mode === 'search' ? (
          // Search Mode
          <div className="search-mode">
            {!hasSearched ? (
              // Initial Search View (Google-like)
              <div className="search-initial">
                <div className="search-logo-container">
                  <h1 className="search-logo">SourceUP</h1>
                  <p className="search-tagline">Find the perfect suppliers for your business</p>
                </div>

                <form onSubmit={handleSearch} className="search-form-initial">
                  <div className="search-input-wrapper">
                    <svg className="search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8"/>
                      <path d="m21 21-4.35-4.35"/>
                    </svg>
                    <input
                      ref={searchInputRef}
                      type="text"
                      className="search-input-initial"
                      placeholder="Search for suppliers... (e.g., LED bulb suppliers in Vietnam)"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                    {searchQuery && (
                      <button type="button" className="clear-btn" onClick={clearSearch}>
                        ×
                      </button>
                    )}
                  </div>
                  <div className="search-buttons">
                    <button type="submit" className="search-btn">
                      Search Suppliers
                    </button>
                    <button type="button" className="lucky-btn" onClick={handleSearch}>
                      I'm Feeling Lucky
                    </button>
                  </div>
                </form>

                <div className="search-examples">
                  <p>Try searching for:</p>
                  <div className="example-chips">
                    <button onClick={() => setSearchQuery('LED bulbs in Vietnam under $2')}>LED bulbs in Vietnam</button>
                    <button onClick={() => setSearchQuery('ISO certified textile suppliers')}>ISO certified textiles</button>
                    <button onClick={() => setSearchQuery('electronics manufacturers in China')}>Electronics in China</button>
                  </div>
                </div>
              </div>
            ) : (
              // Search Results View
              <div className="search-results-view">
                <div className="search-header-compact">
                  <h2 className="logo-compact">SourceUP</h2>
                  <form onSubmit={handleSearch} className="search-form-compact">
                    <div className="search-input-wrapper-compact">
                      <svg className="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8"/>
                        <path d="m21 21-4.35-4.35"/>
                      </svg>
                      <input
                        type="text"
                        className="search-input-compact"
                        placeholder="Search suppliers..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                      />
                      {searchQuery && (
                        <button type="button" className="clear-btn-compact" onClick={clearSearch}>
                          ×
                        </button>
                      )}
                      <button type="submit" className="search-submit-btn">Search</button>
                    </div>
                  </form>
                </div>

                <div className="results-container">
                  {isSearching ? (
                    <div className="loading">
                      <div className="spinner"></div>
                      <p>Searching suppliers...</p>
                    </div>
                  ) : searchResults.length > 0 ? (
                    <>
                      <div className="results-info">
                        About {searchResults.length} results
                      </div>
                      <div className="results-list">
                        {searchResults.map((result, index) => (
                          <div key={index} className="result-card">
                            <div className="result-header">
                              <h3 className="result-title">{result.supplier}</h3>
                              <span className="result-score">Score: {(result.score * 100).toFixed(0)}%</span>
                            </div>
                            <p className="result-product">{result.product}</p>

                            <div className="result-details">
                              {result.price && <span className="detail-item">💰 {result.price}</span>}
                              {result.location && <span className="detail-item">📍 {result.location}</span>}
                              {result.moq && <span className="detail-item">📦 MOQ: {result.moq}</span>}
                              {result.lead_time && <span className="detail-item">⏱️ {result.lead_time}</span>}
                            </div>

                            <div className="result-reasons">
                              {result.reasons.map((reason, idx) => (
                                <span key={idx} className="reason-tag">
                                  ✓ {reason}
                                </span>
                              ))}
                            </div>

                            {result.url && (
                              <a href={result.url} target="_blank" rel="noopener noreferrer" className="view-details-btn">
                                View Details →
                              </a>
                            )}
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="no-results">
                      <p>No suppliers found matching your criteria.</p>
                      <button onClick={clearSearch} className="try-again-btn">Try another search</button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          // Chat Mode
          <div className="chat-mode">
            <div className="chat-container">
              <div className="chat-messages">
                {messages.length === 0 ? (
                  <div className="chat-welcome">
                    <h2>👋 Welcome to SourceBot</h2>
                    <p>Ask me anything about finding suppliers!</p>
                    <div className="chat-suggestions">
                      <button onClick={() => setChatInput("I need LED bulb suppliers in Vietnam")}>
                        I need LED bulb suppliers in Vietnam
                      </button>
                      <button onClick={() => setChatInput("What are the requirements for ISO certification?")}>
                        What are ISO certification requirements?
                      </button>
                      <button onClick={() => setChatInput("Find affordable textile manufacturers")}>
                        Find affordable textile manufacturers
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {messages.map((msg, index) => (
                      <div key={index} className={`message ${msg.role}`}>
                        <div className="message-avatar">
                          {msg.role === 'user' ? '👤' : '🤖'}
                        </div>
                        <div className="message-content">
                          <p>{msg.content}</p>
                          {msg.suppliers && msg.suppliers.length > 0 && (
                            <div className="chat-suppliers">
                              {msg.suppliers.map((supplier, idx) => (
                                <div key={idx} className="chat-supplier-card">
                                  <strong>{supplier.supplier}</strong>
                                  <span className="chat-score">{(supplier.score * 100).toFixed(0)}%</span>
                                  <p>{supplier.product}</p>
                                  <div className="chat-reasons">
                                    {supplier.reasons.map((reason, ridx) => (
                                      <span key={ridx} className="reason-chip">{reason}</span>
                                    ))}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    {isChatLoading && (
                      <div className="message assistant">
                        <div className="message-avatar">🤖</div>
                        <div className="message-content">
                          <div className="typing-indicator">
                            <span></span><span></span><span></span>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </>
                )}
              </div>

              <form onSubmit={handleChatSubmit} className="chat-input-form">
                <input
                  type="text"
                  className="chat-input"
                  placeholder="Ask me about suppliers..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  disabled={isChatLoading}
                />
                <button type="submit" className="chat-send-btn" disabled={isChatLoading || !chatInput.trim()}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="22" y1="2" x2="11" y2="13"/>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                  </svg>
                </button>
              </form>

              {messages.length > 0 && (
                <button onClick={clearChat} className="clear-chat-btn">
                  Clear Chat
                </button>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;