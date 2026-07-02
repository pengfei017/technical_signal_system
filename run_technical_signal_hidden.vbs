Set shell = CreateObject("WScript.Shell")
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
shell.Run "cmd /c ""D:\technical_signal_system\run_technical_signal.bat"" " & args, 0, False
