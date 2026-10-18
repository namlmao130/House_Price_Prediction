from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


SOURCE = Path(r"C:\Users\DELL\Downloads\Báo cáo bài tập nhóm 10 - Dự đoán giá nhà Hà Nội3.docx")
OUTPUT = Path(r"C:\Users\DELL\Desktop\du_doan_gia_nha\output\docx\Bao_cao_nhom_10_Du_doan_gia_nha_Ha_Noi_hoan_thien.docx")
NAVY = "1F4E78"
PALE_BLUE = "EAF2F8"
LIGHT_GRAY = "F5F7FA"
BORDER = "D9D9D9"


def set_run_font(run, name="Times New Roman", size=Pt(12), color="000000"):
    run.font.name = name
    run.font.size = size
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.get_or_add_rFonts()
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), name)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=BORDER, size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    marker = tr_pr.find(qn("w:tblHeader"))
    if marker is None:
        marker = OxmlElement("w:tblHeader")
        tr_pr.append(marker)
    marker.set(qn("w:val"), "true")


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    marker = tr_pr.find(qn("w:cantSplit"))
    if marker is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def replace_paragraph_text(paragraph, text):
    paragraph.clear()
    paragraph.add_run(text)


def replace_all_text(doc, old, new):
    for paragraph in doc.paragraphs:
        if old in paragraph.text:
            replace_paragraph_text(paragraph, paragraph.text.replace(old, new))
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    if old in paragraph.text:
                        replace_paragraph_text(paragraph, paragraph.text.replace(old, new))


def move_after(reference_element, new_element):
    reference_element.addnext(new_element)


def clear_container(container):
    element = container._element
    for child in list(element):
        element.remove(child)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    set_run_font(run, size=Pt(9), color="5B6573")


doc = Document(SOURCE)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.settings.odd_and_even_pages_header_footer = False

# Metadata.
doc.core_properties.title = "Báo cáo bài tập nhóm 10 Dự đoán giá nhà Hà Nội"
doc.core_properties.subject = "Hệ thống phân tích và dự đoán giá nhà Hà Nội bằng học máy"
doc.core_properties.author = "Nhóm 10"
doc.core_properties.keywords = "Python, học máy, dự đoán giá nhà, Hà Nội, Streamlit"

# Page setup and header/footer.
section = doc.sections[0]
section.different_first_page_header_footer = True
section.top_margin = Inches(0.78)
section.bottom_margin = Inches(0.72)
section.left_margin = Inches(0.9)
section.right_margin = Inches(0.9)
section.header_distance = Inches(0.3)
section.footer_distance = Inches(0.3)

clear_container(section.header)
header_p = section.header.add_paragraph()
header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
header_run = header_p.add_run("BÁO CÁO NHÓM 10  |  DỰ ĐOÁN GIÁ NHÀ HÀ NỘI")
set_run_font(header_run, name="Arial", size=Pt(8), color="5B6573")

clear_container(section.footer)
footer_p = section.footer.add_paragraph()
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer_p.add_run("Trang ")
add_field(footer_p, "PAGE")
footer_p.add_run(" / ")
add_field(footer_p, "NUMPAGES")
for run in footer_p.runs:
    set_run_font(run, name="Arial", size=Pt(9), color="5B6573")

clear_container(section.first_page_header)
section.first_page_header.add_paragraph()
clear_container(section.first_page_footer)
section.first_page_footer.add_paragraph()

# Cover page reuses the first three paragraphs and inserts a member table.
p0, p1, p2 = doc.paragraphs[0:3]
replace_paragraph_text(p0, "BÁO CÁO BÀI TẬP NHÓM")
p0.style = doc.styles["Title"]
p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
p0.paragraph_format.space_before = Pt(72)
p0.paragraph_format.space_after = Pt(18)

replace_paragraph_text(p1, "DỰ ĐOÁN GIÁ NHÀ HÀ NỘI BẰNG HỌC MÁY")
p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
p1.paragraph_format.space_after = Pt(44)

replace_paragraph_text(
    p2,
    "NHÓM 10\nMôn học: Python\nGiảng viên: Phạm Danh Tuyên",
)
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.paragraph_format.line_spacing = 1.4
p2.paragraph_format.space_after = Pt(24)

member_table = doc.add_table(rows=1, cols=2)
member_table.alignment = WD_TABLE_ALIGNMENT.CENTER
member_table.autofit = False
member_table.columns[0].width = Inches(3.8)
member_table.columns[1].width = Inches(1.7)
member_table.rows[0].cells[0].text = "Thành viên"
member_table.rows[0].cells[1].text = "MSSV"
for name, student_id in (
    ("Võ Quang Huy", "B24DCKH063"),
    ("Nguyễn Hữu Nam", "B24DCKH092"),
    ("Lê Đình Phong", "B24DCKH098"),
):
    cells = member_table.add_row().cells
    cells[0].text = name
    cells[1].text = student_id
move_after(p2._p, member_table._tbl)

date_p = doc.add_paragraph("Hà Nội, tháng 10 năm 2026")
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_p.paragraph_format.space_before = Pt(34)
move_after(member_table._tbl, date_p._p)

break_p = doc.add_paragraph()
break_p.add_run().add_break(WD_BREAK.PAGE)
move_after(date_p._p, break_p._p)

summary_heading = doc.add_paragraph("Tóm tắt", style="Heading 1")
move_after(break_p._p, summary_heading._p)
summary_p = doc.add_paragraph(
    "Báo cáo trình bày hệ thống dự đoán giá chào bán nhà ở Hà Nội từ 657.691 "
    "mẫu sau làm sạch và khử trùng. Pipeline chia dữ liệu theo nhóm tin trùng, "
    "fit toàn bộ bước tiền xử lý trên train và chọn mô hình bằng GroupKFold 5 fold. "
    "Random Forest được chọn với Test R² 0,711, MAE 5,12 tỷ đồng, Median APE "
    "17,4% và 55,9% dự đoán lệch không quá 20%. Ứng dụng Streamlit cung cấp EDA, "
    "dự đoán giá và báo cáo sai số theo phân khúc."
)
summary_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
move_after(summary_heading._p, summary_p._p)

# Correct and tighten technical wording without changing the report structure.
paragraph_replacements = {
    "Nguồn: bộ dữ liệu tinixai/vietnam-real-estates trên Hugging Face, gồm 10 file parquet":
        "Nguồn dữ liệu: tinixai/vietnam-real-estates trên Hugging Face, gồm 10 file Parquet.",
    "Lọc bỏ loại \"Đất\" (giá đất không phụ thuộc số phòng, số tầng)":
        "Loại bỏ loại \"Đất\" vì bộ đặc trưng hiện tại chưa phù hợp để định giá đất trống; các biến phòng và tầng không áp dụng cho nhóm này.",
    "Ridge (mô hình tuyến tính có điều chuẩn, dễ diễn giải, dùng làm mốc so sánh),":
        "RidgeCV (mô hình tuyến tính có điều chuẩn, dùng làm mốc so sánh),",
    "Random Forest giảm MAE khoảng 58% so với baseline. CV và test gần nhau nên không có dấu hiệu rò rỉ hay quá khớp.":
        "Random Forest giảm MAE khoảng 58% so với baseline. Điểm CV và test không chênh lệch lớn; pipeline đồng thời kiểm soát rò rỉ bằng cách chia theo nhóm và fit preprocessing riêng trong từng fold.",
    "Ridge sau khi log hóa đạt R² log 0,816, không xa hai mô hình cây (0,85): phần lớn cấu trúc của dữ liệu là log-tuyến tính. Các mô hình cây hơn nhờ nắm bắt tương tác, rõ nhất ở R² trên giá gốc (0,711 so với 0,614) và tỷ lệ lệch ≤ 20% (55,9% so với 48,0%).":
        "Ridge sau khi log hóa đạt R² log 0,816, cho thấy mô hình log-tuyến tính giải thích được phần lớn biến thiên giá trên thang log. Hai mô hình cây vẫn tốt hơn nhờ học quan hệ phi tuyến và tương tác giữa các đặc trưng; Random Forest đạt Test R² 0,711 so với 0,614 của Ridge và tỷ lệ lệch không quá 20% là 55,9% so với 48,0%.",
    "Tab 1: \"Tổng quan dữ liệu\" – phân phối giá và giá trung vị theo quận]":
        "Tab 1: \"Tổng quan dữ liệu\" trình bày phân phối giá và giá trung vị theo quận.",
    "Tab 2: \"Dự đoán giá\" – form nhập thông tin và kết quả định giá kèm khoảng tham khảo]":
        "Tab 2: \"Dự đoán giá\" cung cấp biểu mẫu nhập thông tin và kết quả định giá kèm khoảng tham khảo.",
    "Tab 3: \"Đánh giá mô hình\" – bảng so sánh mô hình và sai số theo nhóm]":
        "Tab 3: \"Đánh giá mô hình\" hiển thị bảng so sánh mô hình và sai số theo nhóm.",
    "Dự đoán giá chào bán, không phải giá giao dịch; giá chốt thường thấp hơn.":
        "Hệ thống dự đoán giá chào bán, không phải giá giao dịch; giá giao dịch thực tế có thể khác giá đăng tin.",
    "Yêu cầu Python 3.12 trở lên, chạy ứng dụng không cần huấn luyện lại vì artifact đã có sẵn.":
        "Yêu cầu Python 3.12 trở lên; artifact hiện tại được huấn luyện bằng Python 3.13. Có thể chạy ứng dụng mà không cần huấn luyện lại.",
}
for paragraph in doc.paragraphs:
    replacement = paragraph_replacements.get(paragraph.text.strip())
    if replacement:
        replace_paragraph_text(paragraph, replacement)

for old, new in (
    ("pipline", "pipeline"),
    ("Chuẩn đoán", "Chẩn đoán"),
    ("làm  sạch", "làm sạch"),
    ("Tiền xử lí", "Tiền xử lý"),
    ("chống dataleak", "kiểm soát rò rỉ dữ liệu"),
    ("parquet\\_dataset", "parquet_dataset"),
    ("data\\_loader.py", "data_loader.py"),
    ("Ridge (log1p)", "RidgeCV (log feature)"),
    ("evaluation.py             # MAE/RMSE/R2/MAPE", "evaluation.py             # metric tổng thể và sai số theo nhóm"),
):
    replace_all_text(doc, old, new)

# Promote the heading hierarchy and keep headings with their following content.
for paragraph in doc.paragraphs:
    text = paragraph.text.strip()
    if re.match(r"^\d+\.\d+\s", text):
        paragraph.style = doc.styles["Heading 2"]
    elif re.match(r"^\d+\.\s", text) or text.startswith("Phụ lục") or text == "Tóm tắt":
        paragraph.style = doc.styles["Heading 1"]
    if paragraph.style.name.startswith("Heading"):
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.keep_together = True

# Global style system.
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.paragraph_format.line_spacing = 1.18
normal.paragraph_format.space_after = Pt(4)

title = doc.styles["Title"]
title.font.name = "Times New Roman"
title.font.size = Pt(22)
title.font.bold = True
title.font.color.rgb = RGBColor(0, 0, 0)

for style_name, size, before, after in (
    ("Heading 1", 16, 14, 7),
    ("Heading 2", 13, 10, 5),
    ("Heading 3", 12, 8, 4),
):
    style = doc.styles[style_name]
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)
    style.paragraph_format.keep_with_next = True

if "Code" in [style.name for style in doc.styles]:
    code = doc.styles["Code"]
    code.font.name = "Consolas"
    code.font.size = Pt(9)
    code.font.color.rgb = RGBColor.from_string("1F2937")
    code.paragraph_format.left_indent = Inches(0.18)
    code.paragraph_format.right_indent = Inches(0.12)
    code.paragraph_format.space_before = Pt(4)
    code.paragraph_format.space_after = Pt(6)

list_style = doc.styles["List Paragraph"]
list_style.font.name = "Times New Roman"
list_style.font.size = Pt(11.5)
list_style.paragraph_format.space_after = Pt(2.5)
list_style.paragraph_format.line_spacing = 1.12

# Apply fonts and paragraph layout while preserving emphasis.
for idx, paragraph in enumerate(doc.paragraphs):
    style_name = paragraph.style.name
    if paragraph in (p0, p1, p2, date_p):
        continue
    if style_name == "Code":
        for run in paragraph.runs:
            set_run_font(run, name="Consolas", size=Pt(9), color="1F2937")
    elif style_name.startswith("Heading"):
        size = Pt(16 if style_name == "Heading 1" else 13 if style_name == "Heading 2" else 12)
        for run in paragraph.runs:
            set_run_font(run, size=size)
            run.bold = True
    else:
        size = Pt(11.5 if style_name == "List Paragraph" else 12)
        for run in paragraph.runs:
            set_run_font(run, size=size)
        if style_name == "Normal" and paragraph.text.strip():
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            paragraph.paragraph_format.line_spacing = 1.18

# Cover typography after the global pass.
p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p0.runs:
    set_run_font(run, size=Pt(22))
    run.bold = True
for run in p1.runs:
    set_run_font(run, size=Pt(18))
    run.bold = True
for run in p2.runs:
    set_run_font(run, size=Pt(13))
    run.bold = "NHÓM 10" in run.text
for run in date_p.runs:
    set_run_font(run, size=Pt(11.5))
    run.italic = True

# Highlight lead-in labels without adding boxes.
for paragraph in doc.paragraphs:
    text = paragraph.text.strip()
    if text in {"Nhận xét:", "Kết quả đạt được.", "Hạn chế.", "Hướng phát triển."}:
        for run in paragraph.runs:
            run.bold = True

# Consistent table design.
for table_index, table in enumerate(doc.tables):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    if not table.rows:
        continue
    repeat_header(table.rows[0])
    for row_index, row in enumerate(table.rows):
        prevent_row_split(row)
        for cell_index, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_shading(cell, NAVY if row_index == 0 else (PALE_BLUE if row_index % 2 == 0 else "FFFFFF"))
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.05
                if cell_index > 0 and len(paragraph.text) < 28:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in paragraph.runs:
                    set_run_font(
                        run,
                        name="Arial",
                        size=Pt(9 if len(table.columns) >= 6 else 9.5),
                        color="FFFFFF" if row_index == 0 else "000000",
                    )
                    if row_index == 0:
                        run.bold = True

    # Narrative columns should read like prose rather than centered labels.
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    prose_headers = {
        "Chức năng", "Giải pháp", "Lý do", "Vai trò",
        "Công việc đã thực hiện", "Mục đích sử dụng",
    }
    for column_index, header in enumerate(headers):
        if header in prose_headers:
            for row in table.rows[1:]:
                for paragraph in row.cells[column_index].paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

# Cover table receives a lighter academic style.
set_table_borders(member_table, color="AAB7C4", size="8")
for row_index, row in enumerate(member_table.rows):
    for cell_index, cell in enumerate(row.cells):
        set_cell_shading(cell, NAVY if row_index == 0 else (PALE_BLUE if row_index % 2 == 0 else "FFFFFF"))
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if cell_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                set_run_font(run, name="Arial", size=Pt(10.5), color="FFFFFF" if row_index == 0 else "000000")
                run.bold = row_index == 0

# Keep descriptive text with immediately following tables where practical.
for paragraph in doc.paragraphs:
    next_node = paragraph._p.getnext()
    if next_node is not None and next_node.tag == qn("w:tbl"):
        paragraph.paragraph_format.keep_with_next = True

# Keep the Ridge label and its compact code block together.
for paragraph in doc.paragraphs:
    if paragraph.text.startswith("Pipeline Ridge trên đặc trưng log"):
        paragraph.paragraph_format.keep_with_next = True
    if paragraph.style.name == "Code" and "RidgeCV(alphas" in paragraph.text:
        paragraph.paragraph_format.keep_together = True
    if paragraph.text.startswith("8. Khai báo sử dụng công cụ AI"):
        paragraph.paragraph_format.page_break_before = True

# Add alternative text to the retained screenshot.
for shape in doc.inline_shapes:
    shape._inline.docPr.set("descr", "Bảng đánh giá các mô hình trong ứng dụng Streamlit")

# Avoid widows/orphans and clean direct title colors.
for paragraph in doc.paragraphs:
    ppr = paragraph._p.get_or_add_pPr()
    widow = ppr.find(qn("w:widowControl"))
    if widow is None:
        widow = OxmlElement("w:widowControl")
        ppr.append(widow)
    widow.set(qn("w:val"), "1")

doc.save(OUTPUT)
print(OUTPUT)
