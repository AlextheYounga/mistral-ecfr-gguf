import zipfile

def unzip_file(zip_path):
	# Unzip the file into a json file
	with zipfile.ZipFile(zip_path, "r") as z:
		with z.open(z.namelist()[0]) as f:
			data = f.read()
			# Write the data to a json file
			output_path = zip_path.replace(".zip", "")
			with open(output_path, "wb") as j:
				j.write(data)
			return output_path
	