# Python script to format annotations given by makesense.ai

# Modify paths accordingly
OLD_CSV_PATH = '../../datasets/bb/human/human_unformatted.csv'
NEW_CSV_PATH = '../../datasets/bb/human/human_single.csv'

# Create array for new format
new_csv = [] 

# Open file
print("[INFO] loading file...")
with open(OLD_CSV_PATH) as f:

    print("[INFO] creating new format...")

    # Read lines
    for index, line in enumerate(f):
        
        # Get previous format
        prev_format = line.strip()
        #print("Previous format {}: {}".format(index, prev_format))

        # Strip by comma
        aux = prev_format.split(",")

        # Get end coordinates
        x_end = str(int(aux[1]) + int(aux[3]))
        y_end = str(int(aux[2]) + int(aux[4]))

        # Create new format
        #new_format = aux[5] + "," + aux[1] + "," + aux[2] + "," + x_end + "," + y_end + "," + aux[0]
        new_format = aux[5] + "," + aux[1] + "," + aux[2] + "," + x_end + "," + y_end
        new_csv.append(new_format)
        #print("New format {}: {}".format(index, new_format))

# Save new format to disk
print("[INFO] creating new file...")
f = open(NEW_CSV_PATH, "w")
f.write("\n".join(new_csv))
f.close()
