using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("Fiber Metalens Designer")]
[assembly: AssemblyProduct("Fiber Metalens Designer")]
[assembly: AssemblyDescription("Windows launcher for the fiber metalens design window")]
[assembly: AssemblyVersion("1.0.0.0")]

internal static class FiberMetalensLauncher
{
    [STAThread]
    private static int Main()
    {
        try
        {
            string projectFolder = AppDomain.CurrentDomain.BaseDirectory;
            string designerScript = Path.Combine(projectFolder, "scripts", "fiber_metalens_desktop.py");
            if (!File.Exists(designerScript))
                throw new FileNotFoundException("Keep this executable in the fiber_metalens_nn project folder beside the scripts directory.", designerScript);

            string runtime = Path.Combine(projectFolder, "runtime");
            bool portable = Directory.Exists(runtime);
            string python = portable ? Path.Combine(runtime, "pythonw.exe") : Environment.GetEnvironmentVariable("FIBER_METALENS_PYTHONW");
            if (!portable && String.IsNullOrWhiteSpace(python))
                python = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "anaconda3", "pythonw.exe");
            if (!Path.IsPathRooted(python) || !File.Exists(python))
                throw new FileNotFoundException(portable
                    ? "The bundled Python runtime is missing. Restore the complete distribution folder."
                    : "Set FIBER_METALENS_PYTHONW to the full path of a pythonw.exe with Tk and NumPy.", python);

            var start = new ProcessStartInfo
            {
                FileName = python,
                Arguments = "-B -X utf8 \"" + designerScript + "\"",
                WorkingDirectory = projectFolder,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (portable)
            {
                start.EnvironmentVariables.Remove("PYTHONHOME");
                start.EnvironmentVariables.Remove("PYTHONPATH");
                start.EnvironmentVariables.Remove("VIRTUAL_ENV");
                start.EnvironmentVariables.Remove("CONDA_PREFIX");
                start.EnvironmentVariables["PATH"] = runtime + ";" + Environment.SystemDirectory + ";" + Environment.GetFolderPath(Environment.SpecialFolder.Windows);
                start.EnvironmentVariables["TCL_LIBRARY"] = Path.Combine(runtime, "tcl", "tcl8.6");
                start.EnvironmentVariables["TK_LIBRARY"] = Path.Combine(runtime, "tcl", "tk8.6");
            }
            using (Process designer = Process.Start(start))
            {
                if (designer == null)
                    throw new InvalidOperationException("Windows could not start the designer.");
                if (designer.WaitForExit(1800) && designer.ExitCode != 0)
                    throw new InvalidOperationException("Python exited during startup. Check that this Python runtime has working Tk and NumPy.");
            }
            return 0;
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "Fiber Metalens Designer", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
