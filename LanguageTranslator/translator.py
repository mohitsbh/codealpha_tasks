"""
AI Language Translator using deep-translator
A simple language translator application with GUI interface.
No API key required.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from deep_translator import GoogleTranslator
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES
import threading


class LanguageTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Language Translator")
        self.root.geometry("800x600")
        self.root.configure(bg="#e8edf4")
        
        # Get language list (name -> code mapping)
        self.languages = GOOGLE_LANGUAGES_TO_CODES
        self.lang_list = list(self.languages.keys())
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title Label
        title_label = tk.Label(
            self.root,
            text="🌍 AI Language Translator",
            font=("Helvetica", 26, "bold"),
            bg="#e8edf4",
            fg="#1f2a44"
        )
        title_label.pack(pady=(18, 4))

        subtitle_label = tk.Label(
            self.root,
            text="Fast, free translations with smart search",
            font=("Helvetica", 11),
            bg="#e8edf4",
            fg="#4b5568"
        )
        subtitle_label.pack(pady=(0, 10))
        
        # Main Frame
        main_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief=tk.SOLID, highlightthickness=0)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)
        
        # Left Frame (Source)
        left_frame = tk.Frame(main_frame, bg="#ffffff", padx=8, pady=8)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Source Language Selection
        source_label = tk.Label(
            left_frame,
            text="From:",
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg="#1f2a44"
        )
        source_label.pack(anchor=tk.W)

        # Source language search
        self.source_search_var = tk.StringVar()
        source_search = ttk.Entry(
            left_frame,
            textvariable=self.source_search_var,
            font=("Helvetica", 10)
        )
        source_search.insert(0, "Search languages…")
        source_search.bind("<FocusIn>", lambda e: source_search.delete(0, tk.END) if self.source_search_var.get() == "Search languages…" else None)
        source_search.bind("<FocusOut>", lambda e: source_search.insert(0, "Search languages…") if not self.source_search_var.get() else None)
        source_search.pack(fill=tk.X, pady=(4, 6))
        
        self.source_lang = ttk.Combobox(
            left_frame,
            values=["Auto Detect"] + sorted(self.lang_list),
            state="readonly",
            font=("Helvetica", 11)
        )
        self.source_lang.set("Auto Detect")
        self.source_lang.pack(fill=tk.X, pady=(5, 10))

        # Wire search to filter options
        self.source_search_var.trace_add("write", lambda *args: self._filter_languages(self.source_lang, self.source_search_var.get(), include_auto=True))
        
        # Source Text Area
        source_text_label = tk.Label(
            left_frame,
            text="Enter text to translate:",
            font=("Helvetica", 11),
            bg="#ffffff",
            fg="#1f2a44"
        )
        source_text_label.pack(anchor=tk.W)
        
        self.source_text = scrolledtext.ScrolledText(
            left_frame,
            wrap=tk.WORD,
            font=("Helvetica", 12),
            height=12,
            bg="#fbfcfe",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.source_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Detected Language Label
        self.detected_label = tk.Label(
            left_frame,
            text="",
            font=("Helvetica", 10, "italic"),
            bg="#ffffff",
            fg="#7f8c8d"
        )
        self.detected_label.pack(anchor=tk.W)
        
        # Right Frame (Target)
        right_frame = tk.Frame(main_frame, bg="#ffffff", padx=8, pady=8)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Target Language Selection
        target_label = tk.Label(
            right_frame,
            text="To:",
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg="#1f2a44"
        )
        target_label.pack(anchor=tk.W)

        # Target language search
        self.target_search_var = tk.StringVar()
        target_search = ttk.Entry(
            right_frame,
            textvariable=self.target_search_var,
            font=("Helvetica", 10)
        )
        target_search.insert(0, "Search languages…")
        target_search.bind("<FocusIn>", lambda e: target_search.delete(0, tk.END) if self.target_search_var.get() == "Search languages…" else None)
        target_search.bind("<FocusOut>", lambda e: target_search.insert(0, "Search languages…") if not self.target_search_var.get() else None)
        target_search.pack(fill=tk.X, pady=(4, 6))
        
        self.target_lang = ttk.Combobox(
            right_frame,
            values=sorted(self.lang_list),
            state="readonly",
            font=("Helvetica", 11)
        )
        self.target_lang.set("english")
        self.target_lang.pack(fill=tk.X, pady=(5, 10))

        # Wire search to filter options
        self.target_search_var.trace_add("write", lambda *args: self._filter_languages(self.target_lang, self.target_search_var.get(), include_auto=False))
        
        # Target Text Area
        target_text_label = tk.Label(
            right_frame,
            text="Translation:",
            font=("Helvetica", 11),
            bg="#ffffff",
            fg="#1f2a44"
        )
        target_text_label.pack(anchor=tk.W)
        
        self.target_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=("Helvetica", 12),
            height=12,
            bg="#f8f9fa",
            relief=tk.SOLID,
            borderwidth=1,
            state=tk.DISABLED
        )
        self.target_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Button Frame
        button_frame = tk.Frame(self.root, bg="#e8edf4")
        button_frame.pack(pady=18)
        
        # Translate Button
        self.translate_btn = tk.Button(
            button_frame,
            text="🔄 Translate",
            font=("Helvetica", 14, "bold"),
            bg="#3498db",
            fg="white",
            padx=30,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.translate_text
        )
        self.translate_btn.pack(side=tk.LEFT, padx=10)
        
        # Swap Button
        swap_btn = tk.Button(
            button_frame,
            text="⇄ Swap",
            font=("Helvetica", 12),
            bg="#9b59b6",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.swap_languages
        )
        swap_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear Button
        clear_btn = tk.Button(
            button_frame,
            text="🗑 Clear",
            font=("Helvetica", 12),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.clear_text
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Copy Button
        copy_btn = tk.Button(
            button_frame,
            text="📋 Copy",
            font=("Helvetica", 12),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.copy_translation
        )
        copy_btn.pack(side=tk.LEFT, padx=10)
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            bg="#dce3ef",
            anchor=tk.W,
            padx=10,
            pady=5
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _filter_languages(self, combo_widget, query, include_auto=False):
        """Filter language dropdowns based on search text"""
        query = (query or "").strip().lower()
        # Ignore placeholder text
        if query == "search languages…":
            query = ""

        filtered = [name for name in self.lang_list if query in name.lower()] if query else self.lang_list
        if not filtered:
            filtered = self.lang_list

        values = (["Auto Detect"] if include_auto else []) + sorted(filtered)
        current = combo_widget.get()
        combo_widget["values"] = values

        if current not in values and values:
            combo_widget.set(values[0])
    
    def translate_text(self):
        """Translate the text"""
        source = self.source_text.get("1.0", tk.END).strip()
        
        if not source:
            messagebox.showwarning("Warning", "Please enter text to translate!")
            return
        
        # Disable translate button during translation
        self.translate_btn.config(state=tk.DISABLED)
        self.status_var.set("Translating...")
        
        # Run translation in a separate thread
        thread = threading.Thread(target=self._perform_translation, args=(source,))
        thread.daemon = True
        thread.start()
    
    def _perform_translation(self, source):
        """Perform the actual translation"""
        try:
            # Get source and target language
            src_selection = self.source_lang.get()
            if src_selection == "Auto Detect":
                src_lang = 'auto'
            else:
                src_lang = src_selection.lower()
            
            dest_lang = self.target_lang.get().lower()
            
            # Perform translation
            translator = GoogleTranslator(source=src_lang, target=dest_lang)
            result = translator.translate(source)
            
            # Update UI in main thread
            self.root.after(0, self._update_translation, result, src_lang, dest_lang)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _update_translation(self, result, src_lang, dest_lang):
        """Update the UI with translation result"""
        # Enable target text area and update
        self.target_text.config(state=tk.NORMAL)
        self.target_text.delete("1.0", tk.END)
        self.target_text.insert(tk.END, result)
        self.target_text.config(state=tk.DISABLED)
        
        # Update detected language if auto-detect was used
        if self.source_lang.get() == "Auto Detect":
            self.detected_label.config(text="Language auto-detected")
        else:
            self.detected_label.config(text="")
        
        # Update status and re-enable button
        self.status_var.set(f"Translation complete! ({src_lang} → {dest_lang})")
        self.translate_btn.config(state=tk.NORMAL)
    
    def _show_error(self, error_msg):
        """Show error message"""
        self.status_var.set("Error occurred")
        self.translate_btn.config(state=tk.NORMAL)
        messagebox.showerror("Translation Error", f"An error occurred:\n{error_msg}")
    
    def swap_languages(self):
        """Swap source and target languages"""
        if self.source_lang.get() == "Auto Detect":
            messagebox.showinfo("Info", "Cannot swap when source is set to Auto Detect")
            return
        
        # Swap languages
        src = self.source_lang.get()
        dest = self.target_lang.get()
        self.source_lang.set(dest)
        self.target_lang.set(src)
        
        # Swap text
        src_text = self.source_text.get("1.0", tk.END).strip()
        self.target_text.config(state=tk.NORMAL)
        dest_text = self.target_text.get("1.0", tk.END).strip()
        self.target_text.config(state=tk.DISABLED)
        
        self.source_text.delete("1.0", tk.END)
        self.source_text.insert(tk.END, dest_text)
        
        self.target_text.config(state=tk.NORMAL)
        self.target_text.delete("1.0", tk.END)
        self.target_text.insert(tk.END, src_text)
        self.target_text.config(state=tk.DISABLED)
        
        self.status_var.set("Languages swapped")
    
    def clear_text(self):
        """Clear all text areas"""
        self.source_text.delete("1.0", tk.END)
        self.target_text.config(state=tk.NORMAL)
        self.target_text.delete("1.0", tk.END)
        self.target_text.config(state=tk.DISABLED)
        self.detected_label.config(text="")
        self.status_var.set("Cleared")
    
    def copy_translation(self):
        """Copy translation to clipboard"""
        self.target_text.config(state=tk.NORMAL)
        translation = self.target_text.get("1.0", tk.END).strip()
        self.target_text.config(state=tk.DISABLED)
        
        if translation:
            self.root.clipboard_clear()
            self.root.clipboard_append(translation)
            self.status_var.set("Translation copied to clipboard!")
        else:
            messagebox.showinfo("Info", "No translation to copy!")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = LanguageTranslatorApp(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
