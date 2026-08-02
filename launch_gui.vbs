' Silent launcher. Set RVC_ROOT for best results.
Option Explicit
Dim sh, fso, root, pyw, pyc, envRoot, rc
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = root

envRoot = sh.ExpandEnvironmentStrings("%RVC_ROOT%")
If envRoot <> "%RVC_ROOT%" And envRoot <> "" Then
  If fso.FileExists(envRoot & "\runtime\pythonw.exe") Then pyw = envRoot & "\runtime\pythonw.exe"
  If fso.FileExists(envRoot & "\runtime\python.exe") Then pyc = envRoot & "\runtime\python.exe"
End If

If pyc = "" Then pyc = "python"
If pyw = "" Then
  If InStr(LCase(pyc), "python.exe") > 0 Then
    pyw = Replace(pyc, "python.exe", "pythonw.exe")
    If Not fso.FileExists(pyw) Then pyw = pyc
  Else
    pyw = pyc
  End If
End If

rc = sh.Run("""" & pyc & """ """ & root & "\preflight.py""", 1, True)
If rc <> 0 Then
  MsgBox "GUI preflight failed (exit " & rc & "). Run launch_gui.bat for details.", 16, "rvc_pipeline_gui"
  WScript.Quit rc
End If

sh.Run """" & pyw & """ """ & root & "\gui.py""", 0, False
