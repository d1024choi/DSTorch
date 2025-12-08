import sys
import logging

# =============================================================================
# Progress Printing Utilities (Reusable across the codebase)
# =============================================================================

# ANSI color codes for terminal output
ANSI_COLORS = {
    'CYAN': "\033[96m", 'GREEN': "\033[92m", 'YELLOW': "\033[93m",
    'MAGENTA': "\033[95m", 'RED': "\033[91m", 'BLUE': "\033[94m",
    'BOLD': "\033[1m", 'DIM': "\033[2m", 'RESET': "\033[0m"
}


def progress_bar(current, total, bar_len=25):
    """
    Generate a colored progress bar string.
    
    Args:
        current: Current step (0-indexed)
        total: Total number of steps
        bar_len: Length of the progress bar (default 25)
    
    Returns:
        Formatted progress bar string with color
    
    Example:
        >>> print(progress_bar(50, 100))
        │█████████████░░░░░░░░░░░░│  51.0%
    """
    C = ANSI_COLORS
    progress = (current + 1) / total
    filled = int(bar_len * progress)
    bar = '█' * filled + '░' * (bar_len - filled)
    pct = progress * 100
    color = C['GREEN'] if pct > 66 else (C['YELLOW'] if pct > 33 else C['CYAN'])
    return f"{color}│{bar}│{C['RESET']} {C['BOLD']}{pct:5.1f}%{C['RESET']}"


def format_metric(label, value, fmt='.4f', color='GREEN'):
    """
    Format a single metric with label and colored value.
    
    Args:
        label: Display name for the metric
        value: The value to display
        fmt: Format string (e.g., '.4f', '.2e') or callable
        color: Color key from ANSI_COLORS
    
    Returns:
        Tuple of (colored_string, plain_string)
    
    Example:
        >>> colored, plain = format_metric('Loss', 0.1234, '.4f', 'MAGENTA')
        >>> print(colored)  # Loss:0.1234 (with magenta color)
    """
    C = ANSI_COLORS
    formatted_val = f"{value:{fmt}}" if isinstance(fmt, str) else fmt(value)
    return (
        f"{label}:{C[color]}{formatted_val}{C['RESET']}",  # colored
        f"{label}:{formatted_val}"  # plain
    )


def format_metrics(metrics):
    """
    Format multiple metrics for display.
    
    Args:
        metrics: List of metric dicts, each containing:
            - 'label': str (display name)
            - 'value': the value to display
            - 'fmt': format string (default '.4f') or callable
            - 'color': color key (default 'GREEN')
    
    Returns:
        Tuple of (colored_string, plain_string) joined with separators
    
    Example:
        >>> metrics = [
        ...     {'label': 'Loss', 'value': 0.1234, 'fmt': '.4f', 'color': 'MAGENTA'},
        ...     {'label': 'LR', 'value': 0.001, 'fmt': '.2e', 'color': 'CYAN'},
        ... ]
        >>> colored, plain = format_metrics(metrics)
    """
    C = ANSI_COLORS
    colored_parts, plain_parts = [], []
    for m in metrics:
        label = m['label']
        value = m['value']
        fmt = m.get('fmt', '.4f')
        color = m.get('color', 'GREEN')
        colored, plain = format_metric(label, value, fmt, color)
        colored_parts.append(colored)
        plain_parts.append(plain)
    sep_colored = f" {C['DIM']}│{C['RESET']} "
    sep_plain = " │ "
    return sep_colored.join(colored_parts), sep_plain.join(plain_parts)


def format_metrics_inline(metrics):
    """
    Format metrics for inline progress display (space-separated).
    
    Args:
        metrics: List of metric dicts (same format as format_metrics)
    
    Returns:
        Colored string for terminal display
    """
    C = ANSI_COLORS
    parts = []
    for m in metrics:
        label = m['label']
        value = m['value']
        fmt = m.get('fmt', '.4f')
        color = m.get('color', 'GREEN')
        formatted_val = f"{value:{fmt}}" if isinstance(fmt, str) else fmt(value)
        parts.append(f"{label}:{C[color]}{formatted_val}{C['RESET']}")
    return ' '.join(parts)


def print_training_info(args, logger, tar_dict=None):
    """Print training configuration with fancy formatting.
    
    Args:
        args: Arguments object or dictionary containing configuration values
        logger: Logger object for file output
        tar_dict: Optional dictionary mapping args keys to display configs.
            If None, will try to get from args.tar_dict
    
    Outputs colored version to terminal and clean version to log file.
    """
    # ANSI color codes (for terminal only)
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    WIDTH = 70
    
    def header(title, icon="", use_color=True):
        """Create a section header."""
        title_str = f" {icon} {title} " if icon else f" {title} "
        padding = WIDTH - len(title_str) - 2
        left_pad = padding // 2
        right_pad = padding - left_pad
        if use_color:
            return f"{CYAN}╔{'═' * left_pad}{BOLD}{title_str}{RESET}{CYAN}{'═' * right_pad}╗{RESET}"
        return f"╔{'═' * left_pad}{title_str}{'═' * right_pad}╗"
    
    def separator(use_color=True):
        if use_color:
            return f"{CYAN}╠{'═' * (WIDTH - 2)}╣{RESET}"
        return f"╠{'═' * (WIDTH - 2)}╣"
    
    def footer(use_color=True):
        if use_color:
            return f"{CYAN}╚{'═' * (WIDTH - 2)}╝{RESET}"
        return f"╚{'═' * (WIDTH - 2)}╝"
    
    def row(label, value, color=GREEN, use_color=True):
        """Create a formatted row with label and value."""
        visible_len = len(f"  • {label}: {value}")
        padding = WIDTH - visible_len - 4
        if use_color:
            content = f"  {DIM}•{RESET} {label}: {color}{BOLD}{value}{RESET}"
            return f"{CYAN}║{RESET}{content}{' ' * max(0, padding)}{CYAN}║{RESET}"
        content = f"  • {label}: {value}"
        return f"║{content}{' ' * max(0, padding)}║"
    
    def section_title(title, use_color=True):
        """Create a subsection title."""
        visible_len = len(f"  ▸ {title}")
        padding = WIDTH - visible_len - 4
        if use_color:
            content = f"  {YELLOW}▸ {title}{RESET}"
            return f"{CYAN}║{RESET}{content}{' ' * max(0, padding)}{CYAN}║{RESET}"
        return f"║  ▸ {title}{' ' * max(0, padding)}║"
    
    def build_lines(args, tar_dict=None, use_color=True):
        """Build all output lines based on tar_dict configuration.
        
        Args:
            args: Arguments object or dictionary containing configuration values
            tar_dict: Dictionary mapping args keys to display configs:
                {
                    'key_name': {
                        'Label': 'Display Label',
                        'Color': COLOR_CONSTANT,
                        'use_color': True/False,
                        'format': Optional format string or callable
                    }
                }
            use_color: Global color flag (overridden by individual use_color in tar_dict)
        """
        lines = []
        
        # Helper to get value from args (handles both dict and object)
        def get_value(key):
            if isinstance(args, dict):
                return args.get(key, None)
            else:
                return getattr(args, key, None)
        
        # Main header
        lines.append("")
        dataset_type = get_value('dataset_type') or 'N/A'
        model_name = get_value('model_name') or 'N/A'
        lines.append(header(f"{dataset_type} / {model_name}", "🚀", use_color))
        
        # Print fields from tar_dict
        if tar_dict:
            # Group fields by section name (preserving order of first appearance)
            sections_dict = {}  # section_name -> list of (key, config) tuples
            section_order = []   # Order of section appearance
            no_section = []      # Items without section
            
            for key, config in tar_dict.items():
                section = config.get('section', None)
                if section:
                    if section not in sections_dict:
                        sections_dict[section] = []
                        section_order.append(section)
                    sections_dict[section].append((key, config))
                else:
                    no_section.append((key, config))
            
            # Helper function to format and add a row
            def add_row(key, config):
                value = get_value(key)
                if value is None:
                    return False
                
                # Handle both 'Label' and 'Lable' (typo variant)
                label = config.get('Label') or config.get('Lable') or key
                color = config.get('Color', GREEN)
                # Handle both boolean and string 'true'/'false'
                field_use_color_val = config.get('use_color', use_color)
                if isinstance(field_use_color_val, str):
                    field_use_color = field_use_color_val.lower() == 'true'
                else:
                    field_use_color = bool(field_use_color_val) if field_use_color_val is not None else use_color
                fmt = config.get('format', None)
                
                # Apply formatting if specified
                if fmt:
                    if callable(fmt):
                        value = fmt(value)
                    else:
                        value = f"{value:{fmt}}"
                
                lines.append(row(label, value, color, field_use_color))
                return True
            
            # Print fields without section first
            if no_section:
                for key, config in no_section:
                    add_row(key, config)
            
            # Print sections in the order they first appeared
            for section_name in section_order:
                section_items = sections_dict[section_name]
                if not section_items:
                    continue
                
                lines.append(separator(use_color))
                lines.append(section_title(section_name, use_color))
                
                for key, config in section_items:
                    add_row(key, config)
        
        lines.append(footer(use_color))
        lines.append("")
        return lines
    
    # Build colored lines for terminal, plain lines for log file
    # tar_dict can be passed as parameter, attribute of args, or in args dict
    if tar_dict is None:
        if isinstance(args, dict):
            tar_dict = args.get('tar_dict', None)
        else:
            tar_dict = getattr(args, 'tar_dict', None)
    
    colored_lines = build_lines(args, tar_dict=tar_dict, use_color=True)
    plain_lines = build_lines(args, tar_dict=tar_dict, use_color=False)
    
    # Temporarily disable console handlers to avoid duplicate output
    # Need to check both root logger AND the passed logger
    root_logger = logging.getLogger()
    
    # Collect console handlers from root logger
    root_console_handlers = [h for h in root_logger.handlers 
                             if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    
    # Collect console handlers from the passed logger
    logger_console_handlers = [h for h in logger.handlers 
                               if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    
    # Remove console handlers temporarily
    for handler in root_console_handlers:
        root_logger.removeHandler(handler)
    for handler in logger_console_handlers:
        logger.removeHandler(handler)
    
    # Print colored to terminal, plain to logger (file only now)
    # Use logger.info() which will only write to file handlers (console handlers removed)
    for colored, plain in zip(colored_lines, plain_lines):
        print(colored)  # Colored output to terminal
        # Strip any remaining ANSI codes from plain text as a safety measure
        import re
        plain_clean = re.sub(r'\033\[[0-9;]*m', '', plain)  # Remove ANSI escape sequences
        logger.info(plain_clean)  # Clean output to log file only
    
    # Restore console handlers
    for handler in root_console_handlers:
        root_logger.addHandler(handler)
    for handler in logger_console_handlers:
        logger.addHandler(handler)


if __name__ == '__main__':
    """Test print_training_info() by running this file directly."""
    import sys
    import os
    
    # Add parent directory to path to import argumentparser
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import argument parser
    import argumentparser as ap
    args = ap.args
    
    # Define color constants (same as in print_training_info function)
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    
    # Create a simple logger for testing
    logger = logging.getLogger('test_print')
    logger.setLevel(logging.INFO)
    
    # Create a file handler (optional - comment out if you don't want log file)
    log_file = 'test_print_training_info.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    
    print("="*80)
    print("Testing print_training_info() with default args")
    print("="*80)
    print()
    
    # Test 1: Without tar_dict (uses default BEV display)
    print("Test 1: Without tar_dict (default behavior)")
    print("-" * 80)
    print_training_info(args, logger)
    print()
    
    # Test 2: With tar_dict (custom display)
    print("="*80)
    print("Test 2: With custom tar_dict")
    print("="*80)
    print()
    
    # Example tar_dict demonstrating section grouping
    # Items with the same section name will be grouped together
    # Sections are printed in the order they first appear
    tar_dict = {
        # Items without section (printed first)
        'exp_id': {
            'Label': 'Experiment ID',
            'Color': GREEN,
            'use_color': True
        },
        'gpu_num': {
            'Label': 'GPU Number',
            'Color': GREEN,
            'use_color': True
        },
        'num_epochs': {
            'Label': 'Epochs',
            'Color': GREEN,
            'use_color': True
        },
        'batch_size': {
            'Label': 'Batch Size',
            'Color': GREEN,
            'use_color': True
        },
        
        # First section: "Basic Configuration" (all items with this section grouped together)
        'past_horizon_seconds': {
            'Label': 'Past Horizon',
            'Color': GREEN,
            'use_color': True,
            'format': lambda x: f"{x:.1f} sec",
            'section': 'Basic Configuration'
        },
        'future_horizon_seconds': {
            'Label': 'Future Horizon',
            'Color': GREEN,
            'use_color': True,
            'format': lambda x: f"{x:.1f} sec",
            'section': 'Basic Configuration'
        },
        'target_sample_period': {
            'Label': 'Sample Period',
            'Color': GREEN,
            'use_color': True,
            'format': lambda x: f"{x:.1f} Hz",
            'section': 'Basic Configuration'
        },
        
        # Second section: "Optimizer Settings" (all items with this section grouped together)
        'optimizer_type': {
            'Label': 'Optimizer',
            'Color': MAGENTA,
            'use_color': True,
            'section': 'Optimizer Settings'
        },
        'learning_rate': {
            'Label': 'Learning Rate',
            'Color': GREEN,
            'use_color': True,
            'format': '.5f',
            'section': 'Optimizer Settings'
        },
        'weight_decay': {
            'Label': 'Weight Decay',
            'Color': GREEN,
            'use_color': True,
            'format': '.8f',
            'section': 'Optimizer Settings'
        },
        'apply_lr_scheduling': {
            'Label': 'LR Scheduling',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'Optimizer Settings'
        },
        'lr_schd_type': {
            'Label': 'LR Schedule Type',
            'Color': YELLOW,
            'use_color': True,
            'section': 'Optimizer Settings'
        },
        
        # Third section: "BEV Settings" (demonstrates multiple sections)
        'bev_backbone_type': {
            'Label': 'Backbone',
            'Color': GREEN,
            'use_color': True,
            'section': 'BEV Settings'
        },
        'main_tf_iter': {
            'Label': 'TF Iterations',
            'Color': GREEN,
            'use_color': True,
            'section': 'BEV Settings'
        },
        'init_qmap_size': {
            'Label': 'Init QMap Size',
            'Color': GREEN,
            'use_color': True,
            'section': 'BEV Settings'
        },
        'bool_apply_img_aug': {
            'Label': 'Image Augmentation',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'BEV Settings'
        },
    }
    
    print_training_info(args, logger, tar_dict=tar_dict)
    
    print()
    print("="*80)
    print(f"Test completed! Log file saved to: {log_file}")
    print("="*80)


