' Yoink desktop launcher (PyWebView native window).
' Double-click to start. Closing the window stops the server.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw  = scriptDir & "\.venv\Scripts\pythonw.exe"
launcher = scriptDir & "\desktop.py"

If Not fso.FileExists(pythonw) Then
    MsgBox "Virtual environment not found:" & vbCrLf & pythonw & vbCrLf & vbCrLf & _
           "Create it first: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt", _
           vbCritical, "Yoink"
    WScript.Quit 1
End If

If Not fso.FileExists(launcher) Then
    MsgBox "Launcher not found:" & vbCrLf & launcher, vbCritical, "Yoink"
    WScript.Quit 1
End If

shell.CurrentDirectory = scriptDir
shell.Run """" & pythonw & """ """ & launcher & """", 0, False
