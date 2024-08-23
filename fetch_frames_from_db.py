import psycopg2
import binascii

# Database connection parameters
DB_NAME = 'octopix'
USER = 'postgres'
PASSWORD = '123'
HOST = '192.168.30.159'
PORT = '5432'

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)
cursor = conn.cursor()

# SQL query to retrieve hex-encoded binary data
# cursor.execute("SELECT frame_id, encode(frame, 'hex') AS frame_hex FROM public.camerafeed "
#               "where camera_id = 4 "
#               "and capture_time between '2024-08-07 17:25:05.00072+05:30' and '2024-08-07 17:25:27.00072+05:30' "
#               "ORDER BY frame_id DESC, capture_time DESC LIMIT 500;")
cursor.execute("select frame_id, "
               "encode(frame, 'hex')"
               " AS frame_hex FROM public.camerafeed where frame_id = 30716416"

               )

rows = cursor.fetchall()

# Loop through each row and save the frame as an image
for row in rows:
    frame_id = row[0]
    frame_hex = row[1]

    # Decode the hex string back to binary data
    frame_binary = binascii.unhexlify(frame_hex)

    # Specify the path and filename where you want to save the image
    output_file_path = f'/home/xactai/Desktop/frames_db/frame_{frame_id}.png'  # Change the path and extension as needed

    # Write the binary data to a file
    with open(output_file_path, 'wb') as f:
        f.write(frame_binary)

    print(f"Frame {frame_id} saved as {output_file_path}")

# Close the database connection
cursor.close()
conn.close()

