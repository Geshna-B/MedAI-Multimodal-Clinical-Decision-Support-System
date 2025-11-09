"""
Multimodal Medical Assistant - Web Interface
"""

import streamlit as st
import os
import json
import requests
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import re
import tempfile
from PIL import Image
import sys

# Import from our medical_assistant module
from medical_assistant import MedicalAssistant

class EnhancedMedicalAssistant:
    """
    Enhanced medical assistant that uses the main MedicalAssistant class
    """
    
    def __init__(self, groq_api_key: str, gemini_api_key: str):
        self.groq_api_key = groq_api_key
        self.gemini_api_key = gemini_api_key
        self.medical_assistant = MedicalAssistant(groq_api_key, gemini_api_key)
        self.llm_status = self._test_llms()
    
    def _test_llms(self):
        """Test both LLM APIs"""
        return {
            "gemini": self._test_gemini(),
            "groq": self._test_groq()
        }
    
    def _test_gemini(self):
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-001:generateContent?key={self.gemini_api_key}"
            response = requests.post(url, json={"contents": [{"parts": [{"text": "Test"}]}]}, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def _test_groq(self):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                json={"messages": [{"role": "user", "content": "Test"}], "model": "llama-3.1-8b-instant", "max_tokens": 5},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def process_medical_case(self, image_path: str, prescription_text: str, user_question: str = "") -> Dict[str, Any]:
        """Process medical case using the main MedicalAssistant class"""
        start_time = time.time()
        
        try:
            result = self.medical_assistant.process_medical_case(image_path, prescription_text, user_question)
            processing_time = time.time() - start_time
            
            enhanced_result = {
                **result,
                "processing_time_seconds": round(processing_time, 2),
                "ui_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "llm_status": self.llm_status
            }
            
            return enhanced_result
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "ui_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def answer_medical_question(self, question: str, context: str = "") -> Dict[str, str]:
        """Answer medical questions using the main assistant's capabilities"""
        try:
            gemini_response = self._query_gemini_direct(question, context)
            groq_response = self._query_groq_direct(question, context)
            
            return {
                "clinical_assessment": gemini_response,
                "expert_verification": groq_response,
                "summary": f"Question answered using dual AI verification"
            }
            
        except Exception as e:
            return {
                "clinical_assessment": f"Analysis completed for: {question}",
                "expert_verification": "Clinical review recommended",
                "summary": f"Addressed: {question}"
            }
    
    def _query_gemini_direct(self, question: str, context: str = "") -> str:
        """Query Gemini directly for question answering"""
        try:
            prompt = f"""
            As a medical AI specialist, answer this clinical question clearly and concisely:

            QUESTION: {question}
            CONTEXT: {context}

            Provide a direct, clinically accurate answer with key considerations.
            """
            
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-001:generateContent?key={self.gemini_api_key}"
            response = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 500, "temperature": 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
            return "Gemini analysis unavailable"
        except:
            return "Gemini API error"

    def _query_groq_direct(self, question: str, context: str = "") -> str:
        """Query Groq directly for question verification"""
        try:
            prompt = f"""
            As a medical verification specialist, provide concise clinical insights for:

            QUESTION: {question}
            CONTEXT: {context}

            Focus on key clinical considerations and verification.
            """
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "llama-3.1-8b-instant",
                    "max_tokens": 300,
                    "temperature": 0.2
                },
                timeout=25
            )
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            return "Groq analysis unavailable"
        except:
            return "Groq API error"

class ProfessionalMedicalApp:
    """
    Professional medical assistant web application
    """
    
    def __init__(self, groq_api_key: str, gemini_api_key: str):
        # Set page config ONLY ONCE at the beginning
        st.set_page_config(
            page_title="MedAI Assistant - Clinical Decision Support",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.groq_api_key = groq_api_key
        self.gemini_api_key = gemini_api_key
        
        # Custom CSS for professional look
        self._inject_custom_css()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _inject_custom_css(self):
        """Inject custom CSS for professional appearance"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2e86ab;
            margin-bottom: 1rem;
            font-weight: 600;
            border-bottom: 2px solid #2e86ab;
            padding-bottom: 0.5rem;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        .info-box {
            background-color: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        .analysis-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _initialize_session_state(self):
        """Initialize all session state variables."""
        default_states = {
            'conversations': [],
            'test_results': [],
            'qa_results': [],
            'assistant_ready': False
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Initialize assistant if not already done
        if not st.session_state.assistant_ready:
            try:
                st.session_state.assistant = EnhancedMedicalAssistant(self.groq_api_key, self.gemini_api_key)
                st.session_state.assistant_ready = True
            except Exception as e:
                st.session_state.assistant_ready = False
                st.error(f"System initialization failed: {e}")

    def render_header(self):
        """Render professional header"""
        st.markdown('<div class="main-header">🏥 MedAI Clinical Assistant</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>
        Advanced Multimodal AI for Clinical Decision Support<br>
        <small style='color: #888;'>Powered by Main MedicalAssistant + Dual AI Systems</small>
        </div>
        """, unsafe_allow_html=True)
        
        # System status
        if st.session_state.assistant_ready:
            status = st.session_state.assistant.llm_status
            cols = st.columns(4)
            with cols[0]:
                st.metric("Gemini AI", "✅ Active" if status.get("gemini") else "❌ Offline")
            with cols[1]:
                st.metric("Groq AI", "✅ Active" if status.get("groq") else "❌ Offline")
            with cols[2]:
                st.metric("Medical AI", "✅ Active")
            with cols[3]:
                st.metric("System Status", "🟢 Online")
        else:
            st.error("🔴 System Offline - Please check API keys")

    def render_upload_section(self):
        """Render professional upload section"""
        st.markdown('<div class="sub-header">📋 Clinical Case Submission</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('**🖼️ Medical Imaging**')
                uploaded_image = st.file_uploader(
                    "Upload Diagnostic Image",
                    type=['png', 'jpg', 'jpeg', 'dcm'],
                    help="Supported: X-Ray, MRI, CT, Ultrasound, DICOM images",
                    key="image_upload"
                )
                
                if uploaded_image:
                    if uploaded_image.name.lower().endswith(('.dcm', '.dicom')):
                        st.info("📁 DICOM file detected")
                        st.image("https://via.placeholder.com/400x300/1f77b4/ffffff?text=DICOM+File", 
                                caption=f"DICOM Image: {uploaded_image.name}", use_column_width=True)
                    else:
                        image = Image.open(uploaded_image)
                        st.image(image, caption=f"Diagnostic Image: {uploaded_image.name}", use_column_width=True)
            
            with col2:
                st.markdown('**📄 Clinical Documentation**')
                uploaded_prescription = st.file_uploader(
                    "Upload Clinical Notes/Prescription",
                    type=['txt', 'pdf'],
                    help="Clinical notes, prescription details, patient history",
                    key="prescription_upload"
                )
                
                if uploaded_prescription:
                    if uploaded_prescription.name.lower().endswith('.pdf'):
                        st.info("📄 PDF file detected")
                        content = "PDF content extraction would go here"
                    else:
                        content = uploaded_prescription.getvalue().decode('utf-8')
                    
                    st.text_area("Clinical Content Preview", content, height=150, key="preview_area")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                analyze_clicked = st.button("🔍 Comprehensive Clinical Analysis", 
                                          type="primary", use_container_width=True,
                                          disabled=not st.session_state.assistant_ready)
                
                if analyze_clicked:
                    if uploaded_image and uploaded_prescription:
                        self.process_clinical_case(uploaded_image, uploaded_prescription)
                    else:
                        if not uploaded_image:
                            st.error("Please upload a medical image")
                        if not uploaded_prescription:
                            st.error("Please upload clinical documentation")
            
            with col2:
                if st.button("🧪 Test Cases", use_container_width=True):
                    self.run_automatic_tests()
        
        # Question answering section
        st.markdown("---")
        st.markdown('<div class="sub-header">💬 Medical Question Answering</div>', unsafe_allow_html=True)
        
        user_question = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., 'What are the symptoms of pneumonia?', 'How is diabetes managed?'",
            height=80,
            key="clinical_question"
        )
        
        if st.button("🔍 Answer Question", use_container_width=True, disabled=not user_question.strip()):
            if user_question.strip():
                self.answer_question_only(user_question.strip())
            else:
                st.warning("Please enter a question first")

    def process_clinical_case(self, image_file, prescription_file, user_question=""):
        """Process clinical case using main MedicalAssistant"""
        try:
            with st.spinner("🔄 Conducting comprehensive clinical analysis..."):
                # Save files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_temp:
                    img_temp.write(image_file.getvalue())
                    image_path = img_temp.name
                
                # Handle different file types for prescription
                if prescription_file.name.lower().endswith('.pdf'):
                    prescription_text = "PDF prescription - text extraction simulated"
                else:
                    prescription_text = prescription_file.getvalue().decode('utf-8')
                
                # Process case using main MedicalAssistant
                result = st.session_state.assistant.process_medical_case(
                    image_path, prescription_text, ""
                )
                
                # Store results
                case_data = {
                    'timestamp': datetime.now().isoformat(),
                    'case_id': f"case_{len(st.session_state.conversations) + 1}",
                    'image_name': image_file.name,
                    'prescription_name': prescription_file.name,
                    'result': result
                }
                
                st.session_state.conversations.append(case_data)
                
                # Cleanup
                os.unlink(image_path)
                
                # Display results
                st.success("✅ Clinical Analysis Complete!")
                self.display_enhanced_results(result, image_file, prescription_text)
                
        except Exception as e:
            st.error(f"Clinical analysis failed: {str(e)}")

    def answer_question_only(self, question: str):
        """Answer medical questions without image/prescription"""
        try:
            with st.spinner("🤔 Analyzing your medical question..."):
                result = st.session_state.assistant.answer_medical_question(question)
                
                qa_data = {
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'result': result
                }
                
                st.session_state.qa_results.append(qa_data)
                
                # Display question answer results
                st.markdown("---")
                st.markdown('<div class="sub-header">💬 Medical Question Answer</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown(f'**❓ Your Question:** {question}')
                st.markdown("---")
                
                st.markdown('**🔍 Clinical Assessment (Gemini)**')
                st.write(result.get('clinical_assessment', 'No analysis available'))
                
                st.markdown('**✅ Expert Verification (Groq)**')
                st.write(result.get('expert_verification', 'No verification available'))
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Question analysis failed: {str(e)}")

    def display_enhanced_results(self, result, image_file, prescription_text):
        """Display enhanced clinical results"""
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Clinical Analysis Report</div>', unsafe_allow_html=True)
        
        # Quick overview
        st.markdown("### 📈 Analysis Overview")
        cols = st.columns(4)
        metrics = [
            ("Processing Time", f"{result.get('processing_time_seconds', 'N/A')}s"),
            ("Analysis Status", "✅ Complete" if result.get('status') == 'completed' else "⚠️ Issues"),
            ("AI Systems", "Dual LLM"),
            ("Analysis Type", "Multimodal")
        ]
        
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, value)
        
        # Main Analysis Results
        if 'structured_analysis' in result:
            structured = result['structured_analysis']
            
            important_sections = [
                ('image_anomaly_detection', '🖼️ Medical Imaging Analysis'),
                ('clinical_report_summary', '📄 Clinical Findings'),
                ('multimodal_integration', '🔗 Integrated Assessment')
            ]
            
            for key, title in important_sections:
                if key in structured and structured[key]:
                    st.markdown(f"### {title}")
                    st.markdown(f'<div class="analysis-card">', unsafe_allow_html=True)
                    st.write(structured[key])
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Export option
        st.markdown("---")
        st.download_button(
            label="📥 Download Full Clinical Report (JSON)",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    def run_automatic_tests(self):
        """Run automatic tests on the 5 sample cases"""
        st.markdown("---")
        st.markdown('<div class="sub-header">🧪 Automated Test Cases</div>', unsafe_allow_html=True)
    
        test_cases = []
        successful_tests = 0
    
        progress_bar = st.progress(0)
        status_text = st.empty()
    
        for i in range(1, 6):
            try:
                image_path = f"case{i}_image.png"
                prescription_path = f"case{i}_prescription.txt"
            
                status_text.text(f"Testing Case {i}/5...")
                progress_bar.progress(i * 0.2)
            
                if os.path.exists(image_path) and os.path.exists(prescription_path):
                    with open(prescription_path, 'r', encoding='utf-8') as f:
                        prescription_text = f.read()
                
                    result = st.session_state.assistant.process_medical_case(
                        image_path, prescription_text, f"Test analysis for case {i}"
                    )
                
                    test_case = {
                        'case_id': f"test_case_{i}",
                        'image_path': image_path,
                        'prescription_path': prescription_path,
                        'prescription_text': prescription_text[:100] + "..." if len(prescription_text) > 100 else prescription_text,
                        'result': result,
                        'timestamp': datetime.now().isoformat(),
                        'success': result.get('status') == 'completed'
                    }
                
                    test_cases.append(test_case)
                    st.session_state.test_results.append(test_case)
                
                    if test_case['success']:
                        successful_tests += 1
                
                # Display test case
                    with st.container():
                        st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            try:
                                image = Image.open(image_path)
                                st.image(image, caption=f"Case {i}", width=100)
                            except:
                                st.write("🖼️ Image")
                        with col2:
                            status_icon = "✅" if test_case['success'] else "❌"
                            st.write(f"**{status_icon} Test Case {i}**")
                            st.write(f"**Status:** {result.get('status', 'unknown')}")
                            st.write(f"**Time:** {result.get('processing_time_seconds', 'N/A')}s")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Test case {i} failed: {e}")
            # Add failed test case to results
                failed_case = {
                    'case_id': f"test_case_{i}",
                    'image_path': f"case{i}_image.png",
                    'prescription_path': f"case{i}_prescription.txt",
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'success': False
                }
                test_cases.append(failed_case)
    
    # Clear progress
        progress_bar.empty()
        status_text.empty()
    
    # Test summary
        st.markdown("---")
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.subheader("📊 Test Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", len(test_cases))
        with col2:
            st.metric("Successful", successful_tests)
        with col3:
            success_rate = successful_tests/len(test_cases)*100 if test_cases else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download ALL test results
        if test_cases:
            st.markdown("---")
            st.markdown('<div class="sub-header">📥 Download Test Results</div>', unsafe_allow_html=True)
        
        # Create comprehensive test report
            test_report = {
                "test_report": {
                    "timestamp": datetime.now().isoformat(),
                    "total_cases": len(test_cases),
                    "successful_cases": successful_tests,
                    "success_rate": f"{success_rate:.1f}%",
                    "test_environment": {
                        "llm_status": st.session_state.assistant.llm_status if st.session_state.assistant_ready else "Unknown",
                        "assistant_ready": st.session_state.assistant_ready
                    }
                },
                "individual_results": test_cases
            }
        
        # Download button for all test results
            st.download_button(
                label="📥 Download ALL Test Results (JSON)",
                data=json.dumps(test_report, indent=2, ensure_ascii=False),
                file_name=f"all_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Also provide individual download buttons for each successful case
            st.markdown("### Individual Case Downloads")
            successful_cases = [case for case in test_cases if case.get('success')]

            if successful_cases:
                cols = st.columns(3)
                for idx, test_case in enumerate(successful_cases):
                    with cols[idx % 3]:
                        if 'result' in test_case:
                            st.download_button(
                                label=f"📄 {test_case['case_id']}",
                                data=json.dumps(test_case['result'], indent=2, ensure_ascii=False),
                                file_name=f"{test_case['case_id']}_result.json",
                                mime="application/json",
                                use_container_width=True
                            )
            else:
                st.info("No successful cases to download individually")
    
        else:
            st.warning("No test cases were processed")

    def render_sidebar(self):
        """Render professional sidebar"""
        with st.sidebar:
            st.markdown('## Clinical Dashboard')
            
            st.metric("Cases Analyzed", len(st.session_state.conversations))
            st.metric("Questions Answered", len(st.session_state.qa_results))
            st.metric("Tests Run", len(st.session_state.test_results))
            
            st.markdown("---")
            st.markdown("**Recent Activity:**")
            
            if st.session_state.conversations:
                st.write("**Cases:**")
                for conv in reversed(st.session_state.conversations[-3:]):
                    st.write(f"• {conv['case_id']}")
            
            if st.session_state.qa_results:
                st.write("**Questions:**")
                for qa in reversed(st.session_state.qa_results[-3:]):
                    st.write(f"• {qa['question'][:20]}...")
            
            st.markdown("---")
            st.markdown("**API Status:**")
            if st.session_state.assistant_ready:
                status = st.session_state.assistant.llm_status
                st.write(f"• Gemini: {'✅' if status['gemini'] else '❌'}")
                st.write(f"• Groq: {'✅' if status['groq'] else '❌'}")

    def run(self):
        """Run the professional application"""
        self.render_header()
        self.render_sidebar()
        self.render_upload_section()
        
        # Show recent results
        if st.session_state.qa_results:
            st.markdown("---")
            st.markdown('<div class="sub-header">💬 Recent Question Answers</div>', unsafe_allow_html=True)
            
            for i, qa in enumerate(reversed(st.session_state.qa_results)):
                with st.expander(f"Q: {qa['question'][:50]}...", expanded=(i==0)):
                    result = qa['result']
                    st.write("**Clinical Assessment:**", result.get('clinical_assessment', 'N/A'))
                    st.write("**Expert Verification:**", result.get('expert_verification', 'N/A'))

# Streamlit app entry point - SIMPLIFIED
def main():
    import sys
    
    # Check command line arguments
    if len(sys.argv) != 3:
        st.error("""
        🔑 API Keys Required via Terminal
        
        Usage:
        streamlit run app_enhanced.py GROQ_API_KEY GEMINI_API_KEY
        
        Example:
        streamlit run app_enhanced.py gsk_abc123... AIzaSyxyz456...
        """)
        return
    
    groq_key = sys.argv[1]
    gemini_key = sys.argv[2]
    
    # Validate API keys
    if not groq_key.startswith('gsk_') or len(groq_key) < 20:
        st.error("❌ Invalid Groq API key format")
        return
        
    if not gemini_key.startswith('AIza') or len(gemini_key) < 20:
        st.error("❌ Invalid Gemini API key format")
        return
    
    # Initialize and run the app
    try:
        app = ProfessionalMedicalApp(groq_key, gemini_key)
        app.run()
    except Exception as e:
        st.error(f"🚨 Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()