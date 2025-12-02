"""
MASt3R-SLAM Windows GUI Launcher
A simple GUI for launching MASt3R-SLAM on Windows
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
from pathlib import Path

class SLAMLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("MASt3R-SLAM Launcher (Windows)")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Process handle
        self.process = None
        self.running = False

        # Setup UI
        self.setup_ui()

        # Set default values
        self.load_defaults()

    def setup_ui(self):
        """Setup the GUI layout"""

        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="MASt3R-SLAM Windows Launcher",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)

        # Main container
        main_container = tk.Frame(self.root, padx=20, pady=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # === Dataset Section ===
        dataset_frame = tk.LabelFrame(main_container, text="Dataset Selection", padx=10, pady=10)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))

        # Dataset path
        tk.Label(dataset_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar()
        dataset_entry = tk.Entry(dataset_frame, textvariable=self.dataset_var, width=60)
        dataset_entry.grid(row=0, column=1, padx=5, pady=5)

        browse_dataset_btn = tk.Button(
            dataset_frame,
            text="Browse...",
            command=self.browse_dataset,
            width=10
        )
        browse_dataset_btn.grid(row=0, column=2, padx=5, pady=5)

        # Preset datasets
        tk.Label(dataset_frame, text="Presets:").grid(row=1, column=0, sticky=tk.W, pady=5)
        preset_frame = tk.Frame(dataset_frame)
        preset_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)

        presets = [
            ("TUM xyz", "datasets/tum/rgbd_dataset_freiburg1_xyz"),
            ("TUM desk", "datasets/tum/rgbd_dataset_freiburg1_desk"),
            ("TUM room", "datasets/tum/rgbd_dataset_freiburg1_room"),
        ]

        for i, (name, path) in enumerate(presets):
            btn = tk.Button(
                preset_frame,
                text=name,
                command=lambda p=path: self.dataset_var.set(p),
                width=12
            )
            btn.grid(row=0, column=i, padx=2)

        # === Config Section ===
        config_frame = tk.LabelFrame(main_container, text="Configuration", padx=10, pady=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # Config file
        tk.Label(config_frame, text="Config File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.config_var = tk.StringVar(value="config/base.yaml")
        config_entry = tk.Entry(config_frame, textvariable=self.config_var, width=60)
        config_entry.grid(row=0, column=1, padx=5, pady=5)

        browse_config_btn = tk.Button(
            config_frame,
            text="Browse...",
            command=self.browse_config,
            width=10
        )
        browse_config_btn.grid(row=0, column=2, padx=5, pady=5)

        # Profile selector
        tk.Label(config_frame, text="Profile:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.profile_var = tk.StringVar(value="balanced")
        profile_dropdown = ttk.Combobox(
            config_frame,
            textvariable=self.profile_var,
            values=["balanced", "fast", "quality", "lightweight", "custom"],
            state="readonly",
            width=20
        )
        profile_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Profile info label
        profile_info = tk.Label(
            config_frame,
            text="Balanced: Recommended for most users (3-7 FPS)",
            fg="#666",
            font=("Arial", 8)
        )
        profile_info.grid(row=2, column=1, sticky=tk.W, padx=5)

        # Update info on profile change
        def update_profile_info(*args):
            profile_descriptions = {
                "fast": "Fast: Real-time preview mode (8-12 FPS, lower accuracy)",
                "balanced": "Balanced: Recommended for most users (3-7 FPS)",
                "quality": "Quality: Maximum accuracy (1-3 FPS, best results)",
                "lightweight": "Lightweight: For low-end hardware (10-15 FPS, minimal)",
                "custom": "Custom: Using config file settings"
            }
            profile_info.config(text=profile_descriptions.get(self.profile_var.get(), ""))

        self.profile_var.trace('w', update_profile_info)

        # === Tabbed Options Section ===
        tab_control = ttk.Notebook(main_container)
        tab_control.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Basic Options Tab
        basic_tab = tk.Frame(tab_control, padx=10, pady=10)
        tab_control.add(basic_tab, text="Basic Options")

        # OSC / Advanced Tab
        osc_tab = tk.Frame(tab_control, padx=10, pady=10)
        tab_control.add(osc_tab, text="OSC / Advanced")

        # === Basic Options (in basic_tab) ===
        options_frame = tk.Frame(basic_tab)
        options_frame.pack(fill=tk.BOTH, expand=True)

        # Checkboxes for options
        self.no_viz_var = tk.BooleanVar(value=True)
        self.no_backend_var = tk.BooleanVar(value=False)  # Changed: threading mode works better
        self.use_threading_var = tk.BooleanVar(value=True)  # Changed: enable by default
        self.save_results_var = tk.BooleanVar(value=False)
        self.quiet_mode_var = tk.BooleanVar(value=False)  # Quiet mode (disable debug output)

        tk.Checkbutton(
            options_frame,
            text="Disable Visualization (--no-viz) [Recommended for Windows]",
            variable=self.no_viz_var
        ).grid(row=0, column=0, sticky=tk.W, pady=2)

        tk.Checkbutton(
            options_frame,
            text="Disable Backend (--no-backend) [Not recommended]",
            variable=self.no_backend_var
        ).grid(row=1, column=0, sticky=tk.W, pady=2)

        tk.Checkbutton(
            options_frame,
            text="Use Threading (--use-threading) [RECOMMENDED - Works best!]",
            variable=self.use_threading_var
        ).grid(row=2, column=0, sticky=tk.W, pady=2)

        tk.Checkbutton(
            options_frame,
            text="Save Results (trajectory and reconstruction)",
            variable=self.save_results_var
        ).grid(row=3, column=0, sticky=tk.W, pady=2)

        tk.Checkbutton(
            options_frame,
            text="Quiet Mode (--quiet) [Disable debug output - cleaner console]",
            variable=self.quiet_mode_var
        ).grid(row=4, column=0, sticky=tk.W, pady=2)

        # Calibration file (optional)
        calib_frame = tk.Frame(options_frame)
        calib_frame.grid(row=5, column=0, sticky=tk.W, pady=5)

        tk.Label(calib_frame, text="Calibration (optional):").pack(side=tk.LEFT)
        self.calib_var = tk.StringVar()
        calib_entry = tk.Entry(calib_frame, textvariable=self.calib_var, width=40)
        calib_entry.pack(side=tk.LEFT, padx=5)

        browse_calib_btn = tk.Button(
            calib_frame,
            text="Browse...",
            command=self.browse_calib,
            width=10
        )
        browse_calib_btn.pack(side=tk.LEFT)

        # === OSC / Advanced Options (in osc_tab) ===
        # OSC Settings
        osc_enable_frame = tk.LabelFrame(osc_tab, text="OSC Streaming", padx=10, pady=10)
        osc_enable_frame.pack(fill=tk.X, pady=(0, 10))

        self.osc_enabled_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            osc_enable_frame,
            text="Enable OSC Streaming (Send SLAM data to TouchDesigner/etc)",
            variable=self.osc_enabled_var
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)

        # OSC IP
        tk.Label(osc_enable_frame, text="OSC IP:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.osc_ip_var = tk.StringVar(value="127.0.0.1")
        osc_ip_entry = tk.Entry(osc_enable_frame, textvariable=self.osc_ip_var, width=20)
        osc_ip_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # OSC Port
        tk.Label(osc_enable_frame, text="OSC Port:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.osc_port_var = tk.StringVar(value="9000")
        osc_port_entry = tk.Entry(osc_enable_frame, textvariable=self.osc_port_var, width=20)
        osc_port_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        # OSC Info
        info_text = "OSC sends camera poses, point clouds, and status to external apps.\n" \
                   "Default: 127.0.0.1:9000 (localhost)\n" \
                   "Messages: /slam/camera/pose, /slam/pointcloud/chunk, /slam/status"
        info_label = tk.Label(osc_enable_frame, text=info_text, justify=tk.LEFT, fg="#666")
        info_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        # Checkpoint Settings
        checkpoint_frame = tk.LabelFrame(osc_tab, text="Checkpoint & Recovery", padx=10, pady=10)
        checkpoint_frame.pack(fill=tk.X, pady=(0, 10))

        self.auto_checkpoint_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            checkpoint_frame,
            text="Enable Auto-Checkpoint (Save progress every N frames)",
            variable=self.auto_checkpoint_var
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        tk.Label(checkpoint_frame, text="Checkpoint Interval:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.checkpoint_interval_var = tk.StringVar(value="100")
        checkpoint_interval_entry = tk.Entry(checkpoint_frame, textvariable=self.checkpoint_interval_var, width=10)
        checkpoint_interval_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        tk.Label(checkpoint_frame, text="frames", fg="#666").grid(row=1, column=2, sticky=tk.W)

        # Logging Settings
        logging_frame = tk.LabelFrame(osc_tab, text="Logging", padx=10, pady=10)
        logging_frame.pack(fill=tk.X, pady=(0, 10))

        self.enable_log_file_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            logging_frame,
            text="Save log file",
            variable=self.enable_log_file_var
        ).grid(row=0, column=0, sticky=tk.W, pady=5)

        tk.Label(logging_frame, text="Log Level:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_dropdown = ttk.Combobox(
            logging_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=15
        )
        log_level_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # === Control Buttons ===
        control_frame = tk.Frame(main_container)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = tk.Button(
            control_frame,
            text="Start SLAM",
            command=self.start_slam,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            control_frame,
            text="Stop SLAM",
            command=self.stop_slam,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(
            control_frame,
            text="Clear Output",
            command=self.clear_output,
            width=15,
            height=2
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # === Status ===
        status_frame = tk.Frame(main_container)
        status_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(status_frame, text="Status:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.status_label = tk.Label(status_frame, text="Ready", fg="#27ae60", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # === Output Console ===
        output_frame = tk.LabelFrame(main_container, text="Output Console", padx=5, pady=5)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=80,
            height=15,
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Consolas", 9),
            insertbackground="white"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def load_defaults(self):
        """Load default values"""
        # Check if default dataset exists
        default_dataset = "datasets/tum/rgbd_dataset_freiburg1_xyz"
        if Path(default_dataset).exists():
            self.dataset_var.set(default_dataset)

        self.log("MASt3R-SLAM Launcher initialized")
        self.log(f"Working directory: {os.getcwd()}")
        self.log("Ready to launch SLAM\n")

    def browse_dataset(self):
        """Browse for dataset directory"""
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            self.dataset_var.set(directory)

    def browse_config(self):
        """Browse for config file"""
        filename = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            self.config_var.set(filename)

    def browse_calib(self):
        """Browse for calibration file"""
        filename = filedialog.askopenfilename(
            title="Select Calibration File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            self.calib_var.set(filename)

    def log(self, message):
        """Add message to output console"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()

    def clear_output(self):
        """Clear output console"""
        self.output_text.delete(1.0, tk.END)

    def update_status(self, text, color="#27ae60"):
        """Update status label"""
        self.status_label.config(text=text, fg=color)
        self.root.update()

    def build_command(self):
        """Build the SLAM command"""
        # Python executable
        python_exe = sys.executable

        # Base command
        cmd = [python_exe, "main.py"]

        # Dataset
        dataset = self.dataset_var.get().strip()
        if not dataset:
            raise ValueError("Dataset path is required")
        cmd.extend(["--dataset", dataset])

        # Config
        config = self.config_var.get().strip()
        if config:
            cmd.extend(["--config", config])

        # Profile
        profile = self.profile_var.get().strip()
        if profile and profile != "custom":
            cmd.extend(["--profile", profile])

        # Options
        if self.no_viz_var.get():
            cmd.append("--no-viz")

        if self.no_backend_var.get():
            cmd.append("--no-backend")

        if self.use_threading_var.get():
            cmd.append("--use-threading")

        if self.quiet_mode_var.get():
            cmd.append("--quiet")

        # Logging options
        if not self.enable_log_file_var.get():
            cmd.append("--no-log-file")

        log_level = self.log_level_var.get().strip()
        if log_level:
            cmd.extend(["--log-level", log_level])

        # Checkpoint options
        if self.auto_checkpoint_var.get():
            checkpoint_interval = self.checkpoint_interval_var.get().strip()
            if checkpoint_interval:
                cmd.extend(["--checkpoint-interval", checkpoint_interval])

        # OSC options
        if self.osc_enabled_var.get():
            cmd.append("--osc-enabled")
            osc_ip = self.osc_ip_var.get().strip()
            osc_port = self.osc_port_var.get().strip()
            if osc_ip:
                cmd.extend(["--osc-ip", osc_ip])
            if osc_port:
                cmd.extend(["--osc-port", osc_port])

        # Calibration
        calib = self.calib_var.get().strip()
        if calib:
            cmd.extend(["--calib", calib])

        return cmd

    def start_slam(self):
        """Start SLAM process"""
        if self.running:
            messagebox.showwarning("Already Running", "SLAM is already running!")
            return

        try:
            # Build command
            cmd = self.build_command()

            # Validate dataset exists
            dataset_path = Path(self.dataset_var.get())
            if not dataset_path.exists():
                messagebox.showerror(
                    "Dataset Not Found",
                    f"Dataset directory does not exist:\n{dataset_path}"
                )
                return

            # Log command
            self.log("="*80)
            self.log("Starting SLAM...")
            self.log(f"Command: {' '.join(cmd)}")
            self.log("="*80 + "\n")

            # Update UI
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.update_status("Running...", "#f39c12")
            self.running = True

            # Start process in background thread
            thread = threading.Thread(target=self.run_slam_process, args=(cmd,), daemon=True)
            thread.start()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.log(f"ERROR: {e}\n")
            messagebox.showerror("Error", f"Failed to start SLAM:\n{e}")
            self.reset_ui()

    def run_slam_process(self, cmd):
        """Run SLAM process and capture output"""
        try:
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.getcwd()
            )

            # Read output line by line
            for line in self.process.stdout:
                if not self.running:
                    break
                self.log(line.rstrip())

            # Wait for process to complete
            self.process.wait()

            # Check exit code
            if self.process.returncode == 0:
                self.log("\n" + "="*80)
                self.log("SLAM completed successfully!")
                self.log("="*80)
                self.update_status("Completed", "#27ae60")
            else:
                self.log("\n" + "="*80)
                self.log(f"SLAM exited with code {self.process.returncode}")
                self.log("="*80)
                self.update_status(f"Failed (code {self.process.returncode})", "#e74c3c")

        except Exception as e:
            self.log(f"\nERROR: {e}\n")
            self.update_status("Error", "#e74c3c")
        finally:
            self.reset_ui()

    def stop_slam(self):
        """Stop SLAM process"""
        if self.process and self.running:
            self.log("\n" + "="*80)
            self.log("Stopping SLAM...")
            self.log("="*80 + "\n")

            self.running = False
            self.process.terminate()

            # Wait a bit for graceful shutdown
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log("Force killing process...")
                self.process.kill()

            self.update_status("Stopped", "#e67e22")
            self.reset_ui()

    def reset_ui(self):
        """Reset UI to initial state"""
        self.running = False
        self.process = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SLAMLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
