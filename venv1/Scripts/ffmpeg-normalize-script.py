#!C:\Users\Ula\SpeechRecognition\SpeechRecognition\venv1\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'ffmpeg-normalize==1.2.0','console_scripts','ffmpeg-normalize'
__requires__ = 'ffmpeg-normalize==1.2.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('ffmpeg-normalize==1.2.0', 'console_scripts', 'ffmpeg-normalize')()
    )
