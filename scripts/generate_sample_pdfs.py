"""Generate sample PDF documents for testing the RAG pipeline."""

from pathlib import Path

from fpdf import FPDF


def create_technical_report(output_path: Path):
    """
    Create a technical report PDF with equipment specifications.
    Contains temperature limits, voltage specs, and operational guidelines.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Industrial Equipment Technical Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, """
This technical report outlines the operational specifications and safety guidelines
for the XR-5000 Industrial Processing Unit. The equipment is designed for high-volume
manufacturing environments and requires strict adherence to temperature and voltage
parameters to ensure safe operation.

Key findings from our assessment indicate that the unit operates within acceptable
parameters under normal conditions, but careful monitoring is required during peak
load periods when temperatures may approach critical thresholds.
""")

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "2. Operating Specifications", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, """
The XR-5000 has been tested extensively under various operating conditions.
The following specifications must be maintained:

- Operating Temperature Range: 15°C to 75°C (optimal: 40-60°C)
- Maximum Temperature: 80°C (CRITICAL - automatic shutdown triggers at 85°C)
- Operating Voltage: 220V-240V AC, 50/60 Hz
- Maximum Current Draw: 32A continuous, 45A peak
- Pressure Rating: 85 PSI maximum operating pressure
- Humidity: 20-80% non-condensing

WARNING: Operating the unit above 80°C for extended periods may cause permanent
damage to internal components and void the warranty.
""")

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "3. Performance Data", ln=True)
    pdf.set_font("Helvetica", "", 10)

    # Create a simple table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 8, "Parameter", border=1)
    pdf.cell(40, 8, "Value", border=1)
    pdf.cell(40, 8, "Limit", border=1)
    pdf.cell(50, 8, "Status", border=1, ln=True)

    pdf.set_font("Helvetica", "", 10)
    data = [
        ("Temperature", "72°C", "80°C", "WARNING"),
        ("Voltage", "235V", "240V", "OK"),
        ("Current", "28A", "32A", "OK"),
        ("Pressure", "78 PSI", "85 PSI", "OK"),
        ("Efficiency", "94.2%", ">90%", "OK"),
    ]
    for row in data:
        pdf.cell(50, 8, row[0], border=1)
        pdf.cell(40, 8, row[1], border=1)
        pdf.cell(40, 8, row[2], border=1)
        pdf.cell(50, 8, row[3], border=1, ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "4. Risk Assessment", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, """
Based on current operating data, we have identified the following risks:

HIGH RISK: Temperature readings of 72°C are approaching the 80°C limit.
If ambient temperature increases or load increases, the unit may exceed safe
operating parameters. Recommend immediate inspection of cooling systems.

MEDIUM RISK: During peak production hours (14:00-18:00), power consumption
spikes have been observed reaching 30A, which is 94% of the continuous limit.

RECOMMENDATION: Install additional cooling capacity and consider load
balancing to distribute power consumption more evenly.
""")

    pdf.output(output_path)
    print(f"Created: {output_path}")


def create_product_spec(output_path: Path):
    """
    Create a product specification PDF with detailed tables.
    Contains pricing, SKUs, and technical specifications.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Product Specification Sheet", ln=True, align="C")
    pdf.cell(0, 8, "Model: CloudServer Pro Series", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Product Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, """
The CloudServer Pro Series is our enterprise-grade server solution designed
for high-availability cloud deployments. Available in three configurations
to meet varying workload requirements.
""")

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Model Comparison", ln=True)

    # Product comparison table
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [35, 45, 45, 45]
    headers = ["Feature", "Pro-100", "Pro-200", "Pro-400"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    specs = [
        ("SKU", "CS-PRO-100", "CS-PRO-200", "CS-PRO-400"),
        ("CPU Cores", "16", "32", "64"),
        ("RAM", "64 GB", "128 GB", "256 GB"),
        ("Storage", "2 TB NVMe", "4 TB NVMe", "8 TB NVMe"),
        ("Network", "10 Gbps", "25 Gbps", "100 Gbps"),
        ("Price (USD)", "$4,999", "$8,999", "$15,999"),
        ("Annual Support", "$499", "$899", "$1,599"),
        ("Power (Watts)", "450W", "650W", "950W"),
        ("Rack Units", "1U", "2U", "2U"),
    ]

    for row in specs:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1, align="C")
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Warranty Information", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, """
All CloudServer Pro models include:
- 3-year hardware warranty
- 24/7 technical support
- Next-business-day replacement
- Software updates for 5 years

Extended warranty options available: 5-year ($1,200) or 7-year ($2,400)
""")

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Order Information", ln=True)
    pdf.set_font("Helvetica", "", 10)

    # Order table
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(40, 8, "SKU", border=1)
    pdf.cell(40, 8, "Lead Time", border=1)
    pdf.cell(50, 8, "Min Order Qty", border=1)
    pdf.cell(40, 8, "Availability", border=1, ln=True)

    pdf.set_font("Helvetica", "", 9)
    orders = [
        ("CS-PRO-100", "2-3 weeks", "1 unit", "In Stock"),
        ("CS-PRO-200", "3-4 weeks", "1 unit", "In Stock"),
        ("CS-PRO-400", "4-6 weeks", "1 unit", "Limited"),
    ]
    for row in orders:
        pdf.cell(40, 7, row[0], border=1)
        pdf.cell(40, 7, row[1], border=1)
        pdf.cell(50, 7, row[2], border=1)
        pdf.cell(40, 7, row[3], border=1, ln=True)

    pdf.output(output_path)
    print(f"Created: {output_path}")


def create_financial_summary(output_path: Path):
    """
    Create a financial summary PDF with extensive tables.
    Contains budget data, expenses, and projections.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Q4 2024 Financial Summary", ln=True, align="C")
    pdf.cell(0, 8, "Department: Engineering Operations", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Budget Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, """
This report summarizes the financial performance of the Engineering Operations
department for Q4 2024. Overall spending was within budget, with some categories
showing variance that requires attention.
""")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Expense Summary by Category", ln=True)

    # Expense table
    pdf.set_font("Helvetica", "B", 9)
    cols = [50, 35, 35, 35, 30]
    pdf.cell(cols[0], 8, "Category", border=1)
    pdf.cell(cols[1], 8, "Budget", border=1, align="C")
    pdf.cell(cols[2], 8, "Actual", border=1, align="C")
    pdf.cell(cols[3], 8, "Variance", border=1, align="C")
    pdf.cell(cols[4], 8, "Status", border=1, align="C", ln=True)

    pdf.set_font("Helvetica", "", 9)
    expenses = [
        ("Cloud Infrastructure", "$125,000", "$118,500", "-$6,500", "UNDER"),
        ("Software Licenses", "$45,000", "$52,300", "+$7,300", "OVER"),
        ("Hardware Purchases", "$80,000", "$78,200", "-$1,800", "UNDER"),
        ("Contractor Services", "$60,000", "$67,500", "+$7,500", "OVER"),
        ("Training & Certs", "$15,000", "$12,800", "-$2,200", "UNDER"),
        ("Travel & Events", "$25,000", "$23,100", "-$1,900", "UNDER"),
        ("Miscellaneous", "$10,000", "$11,200", "+$1,200", "OVER"),
        ("TOTAL", "$360,000", "$363,600", "+$3,600", "OVER"),
    ]
    for row in expenses:
        pdf.cell(cols[0], 7, row[0], border=1)
        pdf.cell(cols[1], 7, row[1], border=1, align="R")
        pdf.cell(cols[2], 7, row[2], border=1, align="R")
        pdf.cell(cols[3], 7, row[3], border=1, align="R")
        pdf.cell(cols[4], 7, row[4], border=1, align="C", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Monthly Breakdown", ln=True)

    # Monthly table
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 8, "Month", border=1)
    pdf.cell(40, 8, "Expenses", border=1, align="C")
    pdf.cell(40, 8, "Budget", border=1, align="C")
    pdf.cell(40, 8, "% of Budget", border=1, align="C", ln=True)

    pdf.set_font("Helvetica", "", 9)
    monthly = [
        ("October 2024", "$115,200", "$120,000", "96.0%"),
        ("November 2024", "$122,800", "$120,000", "102.3%"),
        ("December 2024", "$125,600", "$120,000", "104.7%"),
    ]
    for row in monthly:
        pdf.cell(50, 7, row[0], border=1)
        pdf.cell(40, 7, row[1], border=1, align="R")
        pdf.cell(40, 7, row[2], border=1, align="R")
        pdf.cell(40, 7, row[3], border=1, align="C", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Key Findings", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, """
1. Software Licenses exceeded budget by 16.2% ($7,300) due to unplanned
   enterprise license renewals and new security tool acquisitions.

2. Contractor Services overspend of $7,500 (12.5%) was driven by emergency
   staffing needs for the cloud migration project.

3. Overall budget variance of +1.0% ($3,600) is within acceptable limits
   but trending upward month-over-month.

RECOMMENDATION: Request $15,000 budget increase for Q1 2025 to account
for anticipated software license renewals and ongoing contractor needs.
""")

    pdf.output(output_path)
    print(f"Created: {output_path}")


def main():
    """Generate all sample PDFs."""
    output_dir = Path(__file__).parent.parent / "data" / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    create_technical_report(output_dir / "technical_report.pdf")
    create_product_spec(output_dir / "product_spec.pdf")
    create_financial_summary(output_dir / "financial_summary.pdf")

    print(f"\nAll sample PDFs created in: {output_dir}")


if __name__ == "__main__":
    main()
