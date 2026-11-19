param([Parameter(Mandatory=$true)][string]$DocumentPath,[Parameter(Mandatory=$true)][string]$PdfPath)
$ErrorActionPreference='Stop'
$DocumentPath=[System.IO.Path]::GetFullPath($DocumentPath)
$PdfPath=[System.IO.Path]::GetFullPath($PdfPath)
[void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($PdfPath))
$word=$null
$document=$null
try {
    $word=New-Object -ComObject Word.Application
    $word.Visible=$false
    $word.DisplayAlerts=0
    $document=$word.Documents.Open($DocumentPath,$false,$true,$false)
    $document.Repaginate()
    $document.ExportAsFixedFormat($PdfPath,17)
    Write-Output ('PDF exported; pages=' + $document.ComputeStatistics(2))
} finally {
    if ($null -ne $document) { $document.Close(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($null -ne $word) { $word.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
}
