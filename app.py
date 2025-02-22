import os
import subprocess

if os.name == 'nt':
    pip = os.path.join('.venv', 'Scripts', 'pip.exe')
    exe = os.path.join('.venv', 'Scripts', 'python.exe')
elif os.name == 'posix':
    pip = os.path.join('.venv', 'bin', 'pip')
    exe = os.path.join('.venv', 'bin', 'python')

if not os.path.exists('.venv'):
    subprocess.run(['python', '-m', 'venv', '.venv'])
    subprocess.run([pip, 'install', '-r', 'requirements.txt'])

subprocess.Popen([exe, os.path.join('code', 'APapp.py')])
    
