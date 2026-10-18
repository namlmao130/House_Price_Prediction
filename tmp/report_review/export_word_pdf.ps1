$ErrorActionPreference = "Stop"
$docxPath = "C:\Users\DELL\Desktop\du_doan_gia_nha\output\docx\Bao_cao_nhom_10_Du_doan_gia_nha_Ha_Noi_hoan_thien.docx"
$pdfDir = "C:\Users\DELL\Desktop\du_doan_gia_nha\tmp\report_review\final_render"
$pdfPath = Join-Path $pdfDir "report.pdf"

New-Item -ItemType Directory -Force -Path $pdfDir | Out-Null

$word = $null
$document = $null
try {
    Write-Output "Starting Word"
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    Write-Output "Opening document"
    $document = $word.Documents.Open($docxPath, $false, $true, $false, "", "", $false, "", "", 0, 0, $false, $true, 0, $true)
    Write-Output "Exporting PDF"
    $document.ExportAsFixedFormat($pdfPath, 17)
    Write-Output "Export complete"
    Write-Output $pdfPath
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
    }
    if ($null -ne $word) {
        $word.Quit()
    }
}
