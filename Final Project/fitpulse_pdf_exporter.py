# fitpulse_pdf_exporter.py
# 📄 ENHANCED PDF EXPORTER WITH EMBEDDED CHARTS
# Generate professional PDFs with matplotlib visualizations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime
import string

# Try importing PDF library
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class EnhancedPDFExporter:
    """PDF export with embedded chart visualizations"""
    
    @staticmethod
    def _create_severity_chart(insights) -> str:
        """Create severity distribution pie chart"""
        fig, ax = plt.subplots(figsize=(6, 4))
        sizes = [
            insights['severity_distribution']['high'],
            insights['severity_distribution']['medium'],
            insights['severity_distribution']['low']
        ]
        labels = ['High', 'Medium', 'Low']
        colors = ['#e74c3c', '#f39c12', '#10b981']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Severity Distribution', fontsize=14, fontweight='bold')
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        return temp_file.name
    
    @staticmethod
    def _create_anomaly_by_type_chart(insights) -> str:
        """Create anomalies by data type bar chart"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        types = []
        counts = []
        for dt, info in insights['by_data_type'].items():
            types.append(dt.replace('_', ' ').title())
            counts.append(info['anomaly_count'])
        
        bars = ax.bar(types, counts, color='#7c3aed')
        ax.set_xlabel('Data Type', fontsize=10, fontweight='bold')
        ax.set_ylabel('Anomaly Count', fontsize=10, fontweight='bold')
        ax.set_title('Anomalies by Data Type', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        return temp_file.name
    
    @staticmethod
    def _create_hourly_chart(insights) -> str:
        """Create hourly distribution chart"""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        all_hours = list(range(24))
        all_counts = [0] * 24
        
        for dt, pattern in insights.get('hourly_patterns', {}).items():
            if pattern.get('distribution'):
                for hour, count in pattern['distribution'].items():
                    all_counts[hour] += count
        
        ax.bar(all_hours, all_counts, color='#6d28d9', alpha=0.8)
        ax.set_xlabel('Hour of Day', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Anomalies', fontsize=10, fontweight='bold')
        ax.set_title('Anomaly Distribution by Hour', fontsize=14, fontweight='bold')
        ax.set_xticks(all_hours)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        return temp_file.name
    
    @staticmethod
    def _create_day_of_week_chart(insights) -> str:
        """Create day of week distribution chart"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = {day: 0 for day in days_order}
        
        for dt, pattern in insights.get('day_of_week_patterns', {}).items():
            if pattern.get('distribution'):
                for day, count in pattern['distribution'].items():
                    if day in day_counts:
                        day_counts[day] += count
        
        days = list(day_counts.keys())
        counts = list(day_counts.values())
        
        bars = ax.bar(days, counts, color='#5b21b6', alpha=0.8)
        ax.set_xlabel('Day of Week', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Anomalies', fontsize=10, fontweight='bold')
        ax.set_title('Anomaly Distribution by Day of Week', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        return temp_file.name
    
    @staticmethod
    def export_comprehensive_pdf(insights, anomaly_data) -> bytes:
        """Export comprehensive PDF with embedded visualizations"""
        if not PDF_AVAILABLE:
            return b"PDF generation unavailable. Install fpdf: pip install fpdf"
        
        def remove_special_chars(text):
            """Remove problematic Unicode characters"""
            if not isinstance(text, str):
                return str(text)
            # Remove zero-width characters
            for char in ['\u200d', '\u200c', '\u200b', '\ufeff', '\u2060']:
                text = text.replace(char, '')
            # Remove emojis
            import re
            text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', '', text)
            # Keep only ASCII printable
            printable = set(string.printable)
            cleaned = ''.join(c if c in printable else ' ' for c in text)
            cleaned = ' '.join(cleaned.split())
            try:
                cleaned.encode('latin-1')
                return cleaned
            except:
                return cleaned.encode('ascii', 'ignore').decode('ascii')
        
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # === HEADER ===
            pdf.set_font('Arial', 'B', 24)
            pdf.set_text_color(124, 58, 237)
            pdf.cell(0, 15, 'FitPulse Analytics', 0, 1, 'C')
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Executive Health Report', 0, 1, 'C')
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
            pdf.ln(10)
            
            # === EXECUTIVE SUMMARY ===
            pdf.set_font('Arial', 'B', 16)
            pdf.set_fill_color(124, 58, 237)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, 'EXECUTIVE SUMMARY', 0, 1, 'L', True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, f"Total Anomalies Detected: {insights['summary']['total_anomalies']:,}")
            pdf.multi_cell(0, 6, f"Overall Anomaly Rate: {insights['summary']['overall_anomaly_rate']:.2f}%")
            pdf.multi_cell(0, 6, f"Risk Level: {insights['risk_assessment']['level']}")
            pdf.multi_cell(0, 6, f"Health Score: {insights['health_score']['score']}/100 ({insights['health_score']['grade']})")
            pdf.multi_cell(0, 6, f"Interpretation: {remove_special_chars(insights['health_score']['interpretation'])}")
            pdf.ln(5)
            
            # === CHART 1: SEVERITY DISTRIBUTION ===
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 8, 'Severity Distribution', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            severity_chart = EnhancedPDFExporter._create_severity_chart(insights)
            pdf.image(severity_chart, x=15, w=180)
            os.unlink(severity_chart)
            pdf.ln(5)
            
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, f"High Risk Events: {insights['severity_distribution']['high']:,}")
            pdf.multi_cell(0, 6, f"Medium Risk Events: {insights['severity_distribution']['medium']:,}")
            pdf.multi_cell(0, 6, f"Low Risk Events: {insights['severity_distribution']['low']:,}")
            pdf.ln(5)
            
            # === NEW PAGE ===
            pdf.add_page()
            
            # === CHART 2: ANOMALIES BY DATA TYPE ===
            pdf.set_font('Arial', 'B', 16)
            pdf.set_fill_color(124, 58, 237)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, 'ANOMALIES BY DATA TYPE', 0, 1, 'L', True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            
            type_chart = EnhancedPDFExporter._create_anomaly_by_type_chart(insights)
            pdf.image(type_chart, x=10, w=190)
            os.unlink(type_chart)
            pdf.ln(5)
            
            pdf.set_font('Arial', '', 11)
            for dt, info in insights['by_data_type'].items():
                clean_dt = remove_special_chars(dt.upper())
                pdf.multi_cell(0, 6, f"{clean_dt}: {info['anomaly_count']:,} anomalies ({info['anomaly_percentage']:.1f}%)")
                if info.get('anomaly_types'):
                    for atype in info['anomaly_types'][:3]:
                        clean_type = remove_special_chars(atype['type'])
                        pdf.multi_cell(0, 6, f"  - {clean_type}: {atype['count']:,}")
            
            # === NEW PAGE ===
            pdf.add_page()
            
            # === CHART 3: HOURLY PATTERNS ===
            pdf.set_font('Arial', 'B', 16)
            pdf.set_fill_color(124, 58, 237)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, 'HOURLY PATTERNS', 0, 1, 'L', True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            
            hourly_chart = EnhancedPDFExporter._create_hourly_chart(insights)
            pdf.image(hourly_chart, x=5, w=200)
            os.unlink(hourly_chart)
            pdf.ln(5)
            
            pdf.set_font('Arial', '', 11)
            for dt, pattern in insights.get('hourly_patterns', {}).items():
                if pattern.get('peak_hour') is not None:
                    clean_dt = remove_special_chars(dt.title())
                    pdf.multi_cell(0, 6, f"{clean_dt}: Peak at {pattern['peak_hour']}:00 ({pattern['peak_count']:,} anomalies)")
            pdf.ln(5)
            
            # === CHART 4: DAY OF WEEK PATTERNS ===
            pdf.set_font('Arial', 'B', 16)
            pdf.set_fill_color(124, 58, 237)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, 'DAY OF WEEK PATTERNS', 0, 1, 'L', True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            
            dow_chart = EnhancedPDFExporter._create_day_of_week_chart(insights)
            pdf.image(dow_chart, x=10, w=190)
            os.unlink(dow_chart)
            pdf.ln(5)
            
            pdf.set_font('Arial', '', 11)
            for dt, pattern in insights.get('day_of_week_patterns', {}).items():
                if pattern.get('peak_day'):
                    clean_dt = remove_special_chars(dt.title())
                    clean_day = remove_special_chars(pattern['peak_day'])
                    pdf.multi_cell(0, 6, f"{clean_dt}: Peak on {clean_day} ({pattern['peak_count']:,} anomalies)")
            
            # === NEW PAGE ===
            pdf.add_page()
            
            # === RECOMMENDATIONS ===
            pdf.set_font('Arial', 'B', 16)
            pdf.set_fill_color(124, 58, 237)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, 'KEY RECOMMENDATIONS', 0, 1, 'L', True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            
            pdf.set_font('Arial', '', 11)
            for i, rec in enumerate(insights['recommendations'][:15], 1):
                clean_rec = remove_special_chars(rec)
                try:
                    pdf.multi_cell(0, 6, f"{i}. {clean_rec}")
                except:
                    pdf.multi_cell(0, 6, f"{i}. [Recommendation text encoding error]")
            
            # === FOOTER ===
            pdf.ln(10)
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 5, 'This report was generated by FitPulse Analytics - Health Monitoring System', 0, 1, 'C')
            pdf.cell(0, 5, 'For questions or concerns, please consult with a healthcare professional', 0, 1, 'C')
            
            return pdf.output(dest='S').encode('latin-1')
            
        except Exception as e:
            # Fallback error PDF
            error_pdf = FPDF()
            error_pdf.add_page()
            error_pdf.set_font('Arial', 'B', 16)
            error_pdf.cell(0, 10, 'PDF Generation Error', 0, 1, 'C')
            error_pdf.set_font('Arial', '', 11)
            error_pdf.multi_cell(0, 8, f"An error occurred while generating the PDF: {str(e)[:300]}")
            error_pdf.ln(5)
            error_pdf.multi_cell(0, 8, "Please try using the JSON export option or contact support.")
            return error_pdf.output(dest='S').encode('latin-1')
