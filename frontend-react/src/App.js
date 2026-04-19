import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [mode, setMode] = useState('search'); // 'search' or 'chat'
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchMetadata, setSearchMetadata] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // 🆕 Advanced search constraints (collapsible panel)
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxPrice, setMaxPrice] = useState('');
  const [moqBudget, setMoqBudget] = useState('');
  const [location, setLocation] = useState('');
  const [locationMandatory, setLocationMandatory] = useState(false);
  const [certification, setCertification] = useState('');
  const [maxLeadTime, setMaxLeadTime] = useState('');
  const [minYearsExperience, setMinYearsExperience] = useState('');
  const [enableExplanations, setEnableExplanations] = useState(true);
  const [enableWhatIf, setEnableWhatIf] = useState(false);

  // 🆕 What-if scenarios state
  const [whatIfScenarios, setWhatIfScenarios] = useState(null);
  const [showWhatIf, setShowWhatIf] = useState(false);

  // 🆕 Selected result for detailed explanation
  const [selectedResult, setSelectedResult] = useState(null);

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
    setSelectedResult(null);

    try {
      const requestBody = {
        product: searchQuery,
        max_price: maxPrice ? parseFloat(maxPrice) : null,
        moq_budget: moqBudget ? parseFloat(moqBudget) : null,
        location: location || null,
        location_mandatory: locationMandatory,
        certification: certification || null,
        max_lead_time: maxLeadTime ? parseInt(maxLeadTime) : null,
        min_years_experience: minYearsExperience ? parseInt(minYearsExperience) : null,
        enable_explanations: enableExplanations,
        enable_what_if: enableWhatIf
      };

      const response = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      const data = await response.json();

      // Handle both old and new response formats
      if (Array.isArray(data)) {
        // Old format (backward compatible)
        setSearchResults(data);
        setSearchMetadata(null);
        setWhatIfScenarios(null);
      } else {
        // New SourceUp-X format
        setSearchResults(data.results || []);
        setSearchMetadata(data.metadata || null);
        setWhatIfScenarios(data.what_if_scenarios || null);
      }
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
      setSearchMetadata(null);
      setWhatIfScenarios(null);
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
    setSearchMetadata(null);
    setWhatIfScenarios(null);
    setHasSearched(false);
    setSelectedResult(null);
    searchInputRef.current?.focus();
  };

  const clearChat = () => {
    setMessages([]);
  };

  const resetAdvancedFilters = () => {
    setMaxPrice('');
    setMoqBudget('');
    setLocation('');
    setLocationMandatory(false);
    setCertification('');
    setMaxLeadTime('');
    setMinYearsExperience('');
  };

  const getActiveConstraintsCount = () => {
    let count = 0;
    if (maxPrice) count++;
    if (moqBudget) count++;
    if (location) count++;
    if (certification) count++;
    if (maxLeadTime) count++;
    if (minYearsExperience) count++;
    return count;
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="logo">SourceUP<span className="logo-x">-X</span></h1>
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
                  <h1 className="search-logo">SourceUP<span className="search-logo-x">-X</span></h1>
                  <p className="search-tagline">Explainable Procurement Intelligence for SMEs</p>
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

                  {/* 🆕 Advanced Search Toggle */}
                  <div className="advanced-toggle-container">
                    <button
                      type="button"
                      className="advanced-toggle"
                      onClick={() => setShowAdvanced(!showAdvanced)}
                    >
                      {showAdvanced ? '▼' : '▶'} Advanced Search
                      {getActiveConstraintsCount() > 0 && (
                        <span className="constraint-badge">{getActiveConstraintsCount()}</span>
                      )}
                    </button>
                  </div>

                  {/* 🆕 Advanced Search Panel */}
                  {showAdvanced && (
                    <div className="advanced-search-panel">
                      <div className="advanced-grid">
                        {/* Price Constraints */}
                        <div className="advanced-group">
                          <label>💰 Budget Constraints</label>
                          <input
                            type="number"
                            placeholder="Max price per unit ($)"
                            value={maxPrice}
                            onChange={(e) => setMaxPrice(e.target.value)}
                            step="0.01"
                          />
                          <input
                            type="number"
                            placeholder="Total MOQ budget ($)"
                            value={moqBudget}
                            onChange={(e) => setMoqBudget(e.target.value)}
                            step="1"
                          />
                        </div>

                        {/* Location Preferences */}
                        <div className="advanced-group">
                          <label>📍 Location Preferences</label>
                          <input
                            type="text"
                            placeholder="Preferred location (e.g., China)"
                            value={location}
                            onChange={(e) => setLocation(e.target.value)}
                          />
                          <label className="checkbox-label">
                            <input
                              type="checkbox"
                              checked={locationMandatory}
                              onChange={(e) => setLocationMandatory(e.target.checked)}
                              disabled={!location}
                            />
                            Location is mandatory
                          </label>
                        </div>

                        {/* Quality Requirements */}
                        <div className="advanced-group">
                          <label>✅ Quality Requirements</label>
                          <input
                            type="text"
                            placeholder="Required certification (e.g., ISO 9001)"
                            value={certification}
                            onChange={(e) => setCertification(e.target.value)}
                          />
                          <input
                            type="number"
                            placeholder="Min. years on platform"
                            value={minYearsExperience}
                            onChange={(e) => setMinYearsExperience(e.target.value)}
                            min="0"
                            max="20"
                          />
                        </div>

                        {/* Delivery Constraints */}
                        <div className="advanced-group">
                          <label>🚚 Delivery Constraints</label>
                          <input
                            type="number"
                            placeholder="Max lead time (days)"
                            value={maxLeadTime}
                            onChange={(e) => setMaxLeadTime(e.target.value)}
                            min="1"
                            max="180"
                          />
                        </div>

                        {/* Feature Toggles */}
                        <div className="advanced-group">
                          <label>🔧 Features</label>
                          <label className="checkbox-label">
                            <input
                              type="checkbox"
                              checked={enableExplanations}
                              onChange={(e) => setEnableExplanations(e.target.checked)}
                            />
                            Show decision explanations
                          </label>
                          <label className="checkbox-label">
                            <input
                              type="checkbox"
                              checked={enableWhatIf}
                              onChange={(e) => setEnableWhatIf(e.target.checked)}
                            />
                            Enable what-if scenarios
                          </label>
                        </div>
                      </div>

                      <div className="advanced-actions">
                        <button type="button" onClick={resetAdvancedFilters} className="reset-filters-btn">
                          Reset Filters
                        </button>
                      </div>
                    </div>
                  )}

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
                  <h2 className="logo-compact">SourceUP<span className="logo-x-compact">-X</span></h2>
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

                {/* 🆕 Metadata Bar */}
                {searchMetadata && (
                  <div className="metadata-bar">
                    <div className="metadata-item">
                      <span className="metadata-label">Candidates:</span>
                      <span className="metadata-value">{searchMetadata.total_candidates}</span>
                    </div>
                    <div className="metadata-item">
                      <span className="metadata-label">After Constraints:</span>
                      <span className="metadata-value">{searchMetadata.after_constraints}</span>
                    </div>
                    <div className="metadata-item">
                      <span className="metadata-label">Method:</span>
                      <span className="metadata-value">{searchMetadata.ranking_method?.toUpperCase()}</span>
                    </div>
                    {searchMetadata.filters_applied?.length > 0 && (
                      <div className="metadata-item">
                        <span className="metadata-label">Filters:</span>
                        <span className="metadata-value">{searchMetadata.filters_applied.join(', ')}</span>
                      </div>
                    )}
                  </div>
                )}

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
                        {whatIfScenarios && (
                          <button
                            className="what-if-toggle-btn"
                            onClick={() => setShowWhatIf(!showWhatIf)}
                          >
                            🔮 {showWhatIf ? 'Hide' : 'Show'} What-If Scenarios
                          </button>
                        )}
                      </div>

                      {/* 🆕 What-If Scenarios */}
                      {showWhatIf && whatIfScenarios && (
                        <div className="what-if-container">
                          <h3>🔮 What-If Scenarios</h3>
                          <div className="what-if-tabs">
                            {Object.entries(whatIfScenarios).map(([key, scenario]) => (
                              <WhatIfScenario key={key} scenario={scenario} />
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="results-list">
                        {searchResults.map((result, index) => (
                          <div key={index} className="result-card">
                            <div className="result-header">
                              <div className="result-title-section">
                                <span className="result-rank">#{result.rank || index + 1}</span>
                                <h3 className="result-title">{result.supplier}</h3>
                              </div>
                              <div className="result-scores">
                                <span className="result-score">Score: {(result.score * 100).toFixed(0)}%</span>
                                {result.confidence_score && (
                                  <span className="confidence-score">
                                    Confidence: {(result.confidence_score * 100).toFixed(0)}%
                                  </span>
                                )}
                              </div>
                            </div>
                            <p className="result-product">{result.product}</p>

                            <div className="result-details">
                              {result.price && <span className="detail-item">💰 {result.price}</span>}
                              {result.location && <span className="detail-item">📍 {result.location}</span>}
                              {result.moq && <span className="detail-item">📦 MOQ: {result.moq}</span>}
                              {result.lead_time && <span className="detail-item">⏱️ {result.lead_time}</span>}
                            </div>

                            <div className="result-reasons">
                              {result.reasons?.map((reason, idx) => (
                                <span key={idx} className="reason-tag">
                                  ✓ {reason}
                                </span>
                              ))}
                            </div>

                            {/* 🆕 Decision Trace Toggle */}
                            {result.decision_trace && (
                              <button
                                className="show-explanation-btn"
                                onClick={() => setSelectedResult(selectedResult === index ? null : index)}
                              >
                                {selectedResult === index ? '▼ Hide' : '▶ Show'} Decision Breakdown
                              </button>
                            )}

                            {/* 🆕 Expanded Decision Trace */}
                            {selectedResult === index && result.decision_trace && (
                              <DecisionTracePanel trace={result.decision_trace} />
                            )}

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
          // Chat Mode (unchanged)
          <div className="chat-mode">
            <div className="chat-container">
              <div className="chat-messages">
                {messages.length === 0 ? (
                  <div className="chat-welcome">
                    <h2>👋 Welcome to SourceBot-X</h2>
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
                                    {supplier.reasons?.map((reason, ridx) => (
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

// 🆕 Decision Trace Panel Component
function DecisionTracePanel({ trace }) {
  return (
    <div className="decision-trace-panel">
      <h4>📊 Decision Breakdown</h4>

      <div className="trace-summary">
        <p><strong>Final Score:</strong> {trace.final_score?.toFixed(4)}</p>
      </div>

      <div className="trace-contributions">
        <h5>Score Contributions:</h5>
        <div className="contribution-bars">
          {Object.entries(trace.contributions || {})
            .sort((a, b) => b[1].contribution - a[1].contribution)
            .slice(0, 5)
            .map(([factor, data]) => (
              <div key={factor} className="contribution-bar">
                <div className="contribution-label">
                  <span>{factor.replace(/_/g, ' ')}</span>
                  <span className="contribution-value">+{data.contribution?.toFixed(3)}</span>
                </div>
                <div className="contribution-progress">
                  <div
                    className="contribution-fill"
                    style={{ width: `${(data.contribution / trace.final_score) * 100}%` }}
                  ></div>
                </div>
                <p className="contribution-explanation">{data.explanation}</p>
              </div>
            ))}
        </div>
      </div>

      {trace.constraints && (
        <div className="trace-constraints">
          <h5>🔒 Constraint Checks:</h5>
          {trace.constraints.passed_all ? (
            <p className="constraint-pass">✅ Passed all constraints</p>
          ) : (
            <p className="constraint-fail">⚠️ Some constraints not met</p>
          )}
        </div>
      )}

      {trace.summary && trace.summary.length > 0 && (
        <div className="trace-summary-points">
          <h5>Summary:</h5>
          <ul>
            {trace.summary.map((point, idx) => (
              <li key={idx}>{point}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// 🆕 What-If Scenario Component
function WhatIfScenario({ scenario }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="what-if-scenario">
      <button
        className="scenario-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span>{scenario.scenario}</span>
        <span>{expanded ? '▼' : '▶'}</span>
      </button>

      {expanded && (
        <div className="scenario-content">
          <div className="scenario-changes">
            {scenario.top_10_changes?.slice(0, 5).map((change, idx) => (
              <div key={idx} className="rank-change">
                <span className="supplier-name">{change.supplier}</span>
                <span className="rank-shift">
                  #{change.original_rank} → #{change.new_rank}
                  <span className={change.rank_change > 0 ? 'rank-up' : 'rank-down'}>
                    {change.rank_change > 0 ? '📈' : '📉'} {Math.abs(change.rank_change)}
                  </span>
                </span>
              </div>
            ))}
          </div>

          {scenario.new_top_supplier !== scenario.original_top_supplier && (
            <div className="scenario-highlight">
              <strong>🏆 New Top Supplier:</strong> {scenario.new_top_supplier}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;