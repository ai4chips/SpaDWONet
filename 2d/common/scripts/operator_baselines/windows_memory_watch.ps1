$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

                                                                        
                                                                      
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OperatorMemoryInfo {
    [StructLayout(LayoutKind.Sequential)]
    public struct PerformanceInformation {
        public uint Size;
        public UIntPtr CommitTotal, CommitLimit, CommitPeak;
        public UIntPtr PhysicalTotal, PhysicalAvailable, SystemCache;
        public UIntPtr KernelTotal, KernelPaged, KernelNonpaged, PageSize;
        public uint HandleCount, ProcessCount, ThreadCount;
    }
    [DllImport("psapi.dll", SetLastError = true)]
    public static extern bool GetPerformanceInfo(
        out PerformanceInformation information, uint size);
}
'@

try {
    while ($true) {
        $information = New-Object OperatorMemoryInfo+PerformanceInformation
        $size = [Runtime.InteropServices.Marshal]::SizeOf($information)
        if (-not [OperatorMemoryInfo]::GetPerformanceInfo([ref]$information, $size)) {
            throw "GetPerformanceInfo failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        }
        $pageSize = $information.PageSize.ToUInt64()
        $sample = [ordered]@{
            schema = 1
            time_utc = [DateTime]::UtcNow.ToString('o')
            commit_used_bytes = $information.CommitTotal.ToUInt64() * $pageSize
            commit_limit_bytes = $information.CommitLimit.ToUInt64() * $pageSize
            physical_total_bytes = $information.PhysicalTotal.ToUInt64() * $pageSize
            physical_available_bytes = $information.PhysicalAvailable.ToUInt64() * $pageSize
        }
        [Console]::Out.WriteLine(($sample | ConvertTo-Json -Compress))
        [Console]::Out.Flush()
        Start-Sleep -Milliseconds 1000
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 2
}
