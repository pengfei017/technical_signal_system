Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batchPath = scriptDir & "\run_technical_signal.bat"
args = ""
For Each arg In WScript.Arguments
    If Len(args) > 0 Then
        args = args & " "
    End If
    args = args & """" & arg & """"
Next
If Len(args) = 0 Then
    args = """run"""
End If
cmd = shell.ExpandEnvironmentStrings("%ComSpec%") & " /c " & """" & """" & batchPath & """ " & args & """"
exitCode = shell.Run(cmd, 0, True)
WScript.Quit exitCode
