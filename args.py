import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    # Output dimension
    parser.add_argument('--height', type=int, default=50, help='Height of the output.')
    parser.add_argument('--width', type=int, default=50, help='Width of the output.')
    parser.add_argument('--scale', type=int, default=10, help='Scale of each pixel.')
    
    # Bitmap parameters
    parser.add_argument('--bitmap', type=str, default='inputs_bitmap\houses4.png',  help='Path to the bitmap.')
    parser.add_argument('--N', type=int, default=3, help='Size of pattern patchs.')
    parser.add_argument('--flip', action='store_true', help='Use flipped pattern.')
    parser.add_argument('--rotate', action='store_true', help='Use rotated pattern.')
    parser.add_argument('--save-patterns', action='store_true', help='Save the generated patterns.')
    parser.add_argument('--print-rules', action='store_true', help='Print all adjacency rules.')
    
    # Video
    parser.add_argument('--save-video', action='store_true', help='Record a video frame by frame.')
    parser.add_argument('--video', type=str, default='videos\wfc_gen.avi',  help='Path and name for the video.')

    return parser.parse_args()