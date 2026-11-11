import os

INPUT_DIR  = './performance_test'
OUTPUT_DIR = './performance_test'

FILENAMES = [    
	"cam_1.txt",
    "cam_1_dawn.txt",
    "cam_1_rain.txt",
    "cam_2.txt",
    "cam_2_rain.txt",
    "cam_3.txt",
    "cam_3_rain.txt",
    "cam_4.txt",
    "cam_4_dawn.txt",
    "cam_4_rain.txt",
    "cam_5.txt",
    "cam_5_dawn.txt",
    "cam_5_rain.txt",
    "cam_6.txt",
    "cam_6_snow.txt",
    "cam_7.txt",
    "cam_7_dawn.txt",
    "cam_7_rain.txt",
    "cam_8.txt",
    "cam_9.txt",
    "cam_10.txt",
    "cam_11.txt",
    "cam_12.txt",
    "cam_13.txt",
    "cam_14.txt",
    "cam_15.txt",
    "cam_16.txt",
    "cam_17.txt",
    "cam_18.txt",
    "cam_19.txt",
    "cam_20.txt"
]

def combine():
	submission = open(OUTPUT_DIR + "/" + "track1.txt", "w")
	for filename in FILENAMES:
		try:
			real_filename = INPUT_DIR + "/" + filename
			sub = open(real_filename, "r")
			
			lines = sub.readlines()
			
			# Get raw output
			for line in lines:
				details = line.split('\t')
				str_details = details[0] + ' ' + details[1] + ' ' + details[2] + ' ' + details[3] + ' ' + details[4] + '\n'
				submission.write(str_details)

			sub.close()
		except:
			continue
	submission.close()

if __name__ == '__main__':
	INPUT_DIR  = '/'
	OUTPUT_DIR = '/'
	combine()