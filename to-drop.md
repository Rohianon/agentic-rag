Diagram Structure                                                                 
                                                                                    
  Boxes (6 main components, left-to-right/top-to-bottom flow)                       
                                                                                    
  ---                                                                               
  1. Document Input (dark blue box, top-left)                                       
  📄 PDF Documents                                                                  
  • Technical reports                                                               
  • Product specifications                                                          
  • Financial summaries                                                             
                                                                                    
  ---                                                                               
  2. Visual Ingestion (box with 2 sub-boxes)                                        
  PDF Parser                    Table Extractor                                     
  • PyMuPDF extraction          • GPT-4V vision                                     
  • 150 DPI rendering           • JSON output                                       
  • Table detection             • Summaries                                         
  Arrow: parse() from Input → Parser, extract() from Parser → Table Extractor       
                                                                                    
  ---                                                                               
  3. Smart Chunker (single box)                                                     
  Document Chunker                                                                  
  • Tables as atomic units (never split)                                            
  • Semantic paragraph boundaries                                                   
  • 512 tokens + 50 overlap                                                         
                                                                                    
  ---                                                                               
  4. Hybrid Index (box with 2 side-by-side sub-boxes)                               
  Vector Store                  Metadata Store                                      
  • ChromaDB                    • Table JSON                                        
  • OpenAI embeddings           • Source tracking                                   
  • Cosine similarity           • Page numbers                                      
  Arrow: embed() and store() from Chunker → Index                                   
                                                                                    
  ---                                                                               
  5. Query Flow (highlighted box, different color - maybe purple)                   
  👤 User Query                                                                     
  ↓                                                                                 
  Hybrid Retriever                                                                  
  • Semantic search                                                                 
  • Metadata filtering                                                              
  • Relevance threshold: 0.3                                                        
  • Explainability                                                                  
                                                                                    
  ---                                                                               
  6. Agent Layer (box with 2 connected sub-boxes)                                   
  Reasoning Agent               Policy Guardrails                                   
  • Chain-of-thought            • Temperature: 80°C                                 
  • Citation generation         • Voltage: 250V                                     
  • Value extraction            • Pressure: 100 PSI                                 
  Arrow: check() from Agent → Guardrails                                            
                                                                                    
  ---                                                                               
  7. Output (green/success colored box)                                             
  📊 Structured JSON                                                                
  • summary                                                                         
  • key_findings[]                                                                  
  • extracted_data{}                                                                
  • risk_flags[]                                                                    
  • citations[]                                                                     
                                                                                    
  ---                                                                               
  Flow Arrows                                                                       
                                                                                    
  Input → Parser → Table Extractor → Chunker → Index                                
                                                ↑                                   
  User Query → Retriever ──────────────────────┘                                    
                  ↓                                                                 
             Agent → Guardrails → Output                                            
                                                                                    
  Color Scheme (inspired by reference)                                              
                                                                                    
  - Background: Dark (#1a1a2e)                                                      
  - Ingestion boxes: Navy (#16213e)                                                 
  - Index boxes: Dark blue (#0f3460)                                                
  - Agent boxes: Purple (#533483)                                                   
  - Output: Coral/red (#e94560)                                                     
  - Arrows: Light gray or coral                                                     
  - Text: White                                                                     
                 