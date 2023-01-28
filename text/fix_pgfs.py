"""This scripts fixes the math environment error in pgfs output by matplotlib

\pgftext[x=2.702656in,y=0.397499in,,top]{\color{textcolor}\rmfamily\fontsize{9.000000}{10.800000}\selectfont 0.25}%

"""
import sys
import glob
import re

# filenames = sys.argv[1:]
filenames = glob.glob(pathname=r'**/*.pgf', recursive=True)

regex = r'^(?P<start>\\pgftext\[.*\]\{\\color{textcolor}\\rmfamily\\fontsize{9.000000}{10.800000}\\selectfont\s+(\\ensuremath\{-\})?)(?P<number>\d+(\.\d+)?)(?P<percentage>(\\%)?)\}'

if __name__ == '__main__':
    for filename in filenames:
        lines = []
        output_lines = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            match = re.match(regex, line)
            if match:
                line = re.sub(regex, r'\g<start>\\ensuremath{\g<number>\g<percentage>}}', line)
            output_lines.append(line)
        
        with open(filename, 'w') as f:
            f.writelines(output_lines)

sys.stderr.write(f'Fixed axis labels in {len(filenames)} files.\n')