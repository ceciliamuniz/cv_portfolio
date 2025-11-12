import React, { useState, useEffect } from 'react';
import './TemplateMatching.css';

const TemplateMatching = () => {
  const [results, setResults] = useState(null);
  const [selectedResult, setSelectedResult] = useState(null);
  const [filterQuality, setFilterQuality] = useState('ALL');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load the JSON data (you'll need to place template_matching_results.json in public folder)
    fetch('/template_matching_results.json')
      .then(response => response.json())
      .then(data => {
        setResults(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading template matching results:', error);
        setLoading(false);
      });
  }, []);

  const getFilteredResults = () => {
    if (!results) return [];
    
    if (filterQuality === 'ALL') return results.results;
    
    return results.results.filter(result => 
      result.methods.some(method => method.quality === filterQuality)
    );
  };

  const getBestMethod = (result) => {
    return result.methods.reduce((best, current) => 
      current.confidence > best.confidence ? current : best
    );
  };

  if (loading) {
    return (
      <div className="template-matching">
        <div className="loading">Loading template matching results...</div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="template-matching">
        <div className="error">Failed to load template matching results</div>
      </div>
    );
  }

  return (
    <div className="template-matching">
      <header className="tm-header">
        <h1>Template Matching Results</h1>
        <div className="tm-stats">
          <span>📊 {results.metadata.total_combinations} combinations</span>
          <span>🖼️ {results.metadata.scene_count} scenes</span>
          <span>🎯 {results.metadata.template_count} templates</span>
        </div>
      </header>

      <div className="tm-controls">
        <label>
          Filter by Quality:
          <select 
            value={filterQuality} 
            onChange={(e) => setFilterQuality(e.target.value)}
          >
            <option value="ALL">All Results</option>
            <option value="EXCELLENT">Excellent (&gt;0.8)</option>
            <option value="GOOD">Good (0.6-0.8)</option>
            <option value="FAIR">Fair (0.4-0.6)</option>
            <option value="POOR">Poor (0.2-0.4)</option>
            <option value="VERY POOR">Very Poor (&lt;0.2)</option>
          </select>
        </label>
      </div>

      <div className="tm-results-grid">
        {getFilteredResults().map((result) => {
          const bestMethod = getBestMethod(result);
          return (
            <div 
              key={result.id} 
              className={`tm-result-card ${bestMethod.quality.toLowerCase().replace(' ', '-')}`}
              onClick={() => setSelectedResult(result)}
            >
              <div className="tm-card-header">
                <h3>Combination #{result.id}</h3>
                <span className={`tm-quality-badge ${bestMethod.quality.toLowerCase().replace(' ', '-')}`}>
                  {bestMethod.quality}
                </span>
              </div>
              
              <div className="tm-images">
                <div className="tm-image-container">
                  <img 
                    src={`data:image/jpeg;base64,${result.scene_image}`}
                    alt={`Scene: ${result.scene_name}`}
                  />
                  <label>Scene: {result.scene_name}</label>
                </div>
                
                <div className="tm-arrow">→</div>
                
                <div className="tm-image-container">
                  <img 
                    src={`data:image/jpeg;base64,${result.template_image}`}
                    alt={`Template: ${result.template_name}`}
                  />
                  <label>Template: {result.template_name}</label>
                </div>
              </div>
              
              <div className="tm-best-result">
                <strong>Best Match: {bestMethod.method}</strong>
                <span>Confidence: {bestMethod.confidence}</span>
              </div>
            </div>
          );
        })}
      </div>

      {selectedResult && (
        <div className="tm-modal" onClick={() => setSelectedResult(null)}>
          <div className="tm-modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="tm-close" onClick={() => setSelectedResult(null)}>×</button>
            
            <h2>Detailed Results - Combination #{selectedResult.id}</h2>
            
            <div className="tm-modal-images">
              <div>
                <img src={`data:image/jpeg;base64,${selectedResult.scene_image}`} alt="Scene" />
                <p>Scene: {selectedResult.scene_name}</p>
              </div>
              <div>
                <img src={`data:image/jpeg;base64,${selectedResult.template_image}`} alt="Template" />
                <p>Template: {selectedResult.template_name}</p>
              </div>
            </div>

            <div className="tm-methods-results">
              {selectedResult.methods.map((method, index) => (
                <div key={index} className="tm-method-result">
                  <h3>{method.method}</h3>
                  <div className="tm-method-details">
                    <img 
                      src={`data:image/png;base64,${method.result_image}`} 
                      alt={`${method.method} result`}
                    />
                    <div className="tm-method-stats">
                      <div className={`tm-confidence ${method.quality.toLowerCase().replace(' ', '-')}`}>
                        Confidence: {method.confidence}
                      </div>
                      <div>Quality: {method.quality}</div>
                      <div>Location: ({method.location.x}, {method.location.y})</div>
                      <div>Size: {method.location.width}×{method.location.height}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TemplateMatching;