import subprocess
import sys

if __name__ == "__main__":
	model_name = str(sys.argv[1])

	subprocess.Popen(
	        '''export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so; python load_agent.py '%s' ''' % model_name, shell=True)