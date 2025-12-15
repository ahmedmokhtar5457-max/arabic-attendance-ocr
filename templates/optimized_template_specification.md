# Optimized Arabic Attendance Template Specification

## Design Principles for OCR Optimization

### 1. General Template Properties
- **Language**: Arabic (RTL layout)
- **Page Size**: A4 (210 × 297 mm)
- **Page Width**: 300 mm (for better spacing)
- **Resolution**: 300 DPI minimum
- **Background**: Pure white (#FFFFFF)
- **Text Color**: Pure black (#000000)
- **Margins**: 20mm on all sides

### 2. Table Structure (OCR-Optimized)

#### 2.1 Line Rules
- **Vertical Lines Only**: No horizontal lines between rows
- **Line Weight**: 0.2mm (thin but visible)
- **Line Color**: Black (#000000)
- **No decorative elements**: Clean, minimal design

#### 2.2 Column Layout (Right to Left)
| Column | Arabic Header | Width (mm) | Purpose | OCR Priority |
|--------|---------------|------------|---------|--------------|
| 1 | م | 15 | Index | Low |
| 2 | الاسم | 60 | Name | Low |
| 3 | وقت التسليم | 25 | **Attendance Time** | **HIGH** |
| 4 | نقطة الحراسة | 40 | **Rank/Guard Point** | **HIGH** |
| 5 | الوردية | 25 | **Shift ID** | **HIGH** |
| 6 | التوقيع | 30 | Signature | Low |
| 7 | ملاحظات | 40 | Notes | Low |

#### 2.3 Critical OCR Columns Design

**Attendance Time Column (وقت التسليم)**
- Width: 25mm
- Horizontal padding: 3mm each side
- Expected values: Arabic numerals 1-24
- Font size guide: 12pt minimum
- Cell height: 8mm minimum

**Rank Column (نقطة الحراسة)**
- Width: 40mm
- Horizontal padding: 4mm each side
- Expected values: English words from closed vocabulary
- Font size guide: 10pt minimum
- Cell height: 8mm minimum

**Shift ID Column (الوردية)**
- Width: 25mm
- Horizontal padding: 3mm each side
- Expected values: Arabic numerals
- Font size guide: 12pt minimum
- Cell height: 8mm minimum

### 3. Header Section (Outside Table)

#### 3.1 Document Information
- **الموقع (Location)**: Top right, 40mm from edge
- **التاريخ (Date)**: Top center, format: DD/MM/YYYY
- **Document Number**: Top left corner
- **Year**: Top right corner (2025)

#### 3.2 Header Spacing
- Header height: 30mm from top
- Clear separation from table: 10mm gap

### 4. Table Body Specifications

#### 4.1 Row Design
- **Row Height**: 8mm minimum
- **No horizontal separators**: Rows defined by content only
- **Row Count**: 25-30 rows per page
- **Alternating background**: None (pure white only)

#### 4.2 Cell Specifications
- **Text Alignment**: Center for numbers, right for Arabic text
- **Padding**: 2mm minimum from cell borders
- **Font Guidelines**: 
  - Arabic numerals: Bold, 12pt
  - English text: Regular, 10pt
  - Arabic text: Regular, 11pt

### 5. OCR Enhancement Features

#### 5.1 Spacing Optimization
- **Inter-column spacing**: 2mm minimum
- **Character spacing**: Normal (no condensed fonts)
- **Line spacing**: 1.2x font size

#### 5.2 Contrast Requirements
- **Background**: Pure white (RGB: 255,255,255)
- **Text**: Pure black (RGB: 0,0,0)
- **Lines**: Black, 0.2mm weight
- **No gradients or shadows**

#### 5.3 Print Quality Guidelines
- **Resolution**: 300 DPI minimum for scanning
- **Paper**: White, non-glossy
- **Ink**: Black, high contrast
- **Avoid**: Colored inks, highlighters, correction fluid

### 6. Template Validation Checklist

#### 6.1 Design Validation
- [ ] Vertical lines only
- [ ] Consistent column widths
- [ ] Adequate padding in target columns
- [ ] High contrast (black on white)
- [ ] No decorative elements

#### 6.2 OCR Readiness
- [ ] Target columns clearly separated
- [ ] Sufficient space for handwriting
- [ ] No overlapping elements
- [ ] Clean, minimal design
- [ ] Proper Arabic RTL layout

#### 6.3 Print Test
- [ ] 300 DPI scan quality
- [ ] Clear column boundaries
- [ ] Readable at actual size
- [ ] No bleeding or smudging
- [ ] Consistent line weights

### 7. Differences from Old Template

#### 7.1 Removed Elements
- **National ID fields**: Completely removed for privacy
- **Horizontal row lines**: Removed for cleaner OCR
- **Complex headers**: Simplified structure
- **Decorative borders**: Minimalist approach

#### 7.2 Enhanced Elements
- **Column spacing**: Increased for better separation
- **Target column width**: Optimized for handwriting
- **Contrast**: Pure black/white only
- **Font guidelines**: Specific recommendations

#### 7.3 OCR-Specific Improvements
- **Vertical-only lines**: Easier column detection
- **Consistent spacing**: Predictable layout
- **Minimal design**: Reduced noise
- **Clear target areas**: Better cell isolation

### 8. Implementation Notes

#### 8.1 Template Creation
- Use vector graphics (SVG/PDF) for scalability
- Ensure exact measurements
- Test print at actual size
- Validate with sample handwriting

#### 8.2 Quality Control
- Regular template validation
- Feedback from users
- OCR accuracy monitoring
- Continuous improvement

This specification ensures maximum OCR accuracy while maintaining usability for manual completion.