import requests
import re
import numpy as np
import pandas as pd
import shutil

'''
Author: Devvrat Raghav
Purpose: To extract the Biological Information and images for Pokemon from Bulbapedia.
         This is done in four parts. The first part retrieves the bio and CDN directory links.
         The second part of the script downloads and stores the Pokemon's image.
         The third part creates a vector of booleans for each Pokemon, indicating which of 
         the 20 selected moves are learnt by that Pokemon. 
         The final part combines all this data into one comphrensive file.

'''

# Part 1 - biology and imageurl extraction

# Get list of Pokemon Names
df = pd.read_csv('D:/UIP/scraping/pokemonstats.csv', header=0)
pokemon_names = df['Name']

# Lists to store the biological information and bulbapedia image URL for each Pokemon
bio = []
imageurls = []

for i in range(802):

    # Handling special cases of Pokemon names with different URL structure
    if pokemon_names[i] == 'Nidoran-M':
        URL = "https://bulbapedia.bulbagarden.net/wiki/{}_(Pok%C3%A9mon)".format('Nidoran%E2%99%82')
    elif pokemon_names[i] == 'Nidoran-F':
        URL = "https://bulbapedia.bulbagarden.net/wiki/{}_(Pok%C3%A9mon)".format('Nidoran%E2%99%80')
    else:
        URL = "https://bulbapedia.bulbagarden.net/wiki/{}_(Pok%C3%A9mon)".format(pokemon_names[i])
    
    # Getting HTML data from bulbapedia page
    r = requests.get(URL)

    # Searching for html tags with CDN directory
    imgloc = re.search(r'<img alt="(.*?) src="(.*?)" width="250"', r.text).group(2)

    # Getting CDN sub-directory with Pokemon's image
    details = re.search(r'thumb/(.*?).png', imgloc).group(1)
    imageurls.append(details)

    # Getting the text from the Biology section on Bulbapedia
    content = re.search(
        '<h2><span class="mw-headline" id="Biology">Biology</span></h2>(.*?)<h2><span class="mw-headline" id="In_the_anime">In the anime</span></h2>',
        r.text,
        re.DOTALL
    ).group(1)

    # Removing HTML tags and cleaning text 
    content = re.sub(r'&#.{4};', '', content)
    content = re.sub(r'<a href=(.*?)>', '', content)
    content = re.sub(r'<(/)?(p|sup|a|b|span|I)>', '', content)
    content = re.sub(r'\(Japanese:(.*?)\)', '', content)
    content = re.sub(r'<(span) class(.*?)>', '', content)
    content = re.sub(r'<img (.*)/>', '', content)
    content = re.sub(r'<sup id(.*?)>', '', content)
    content = re.sub(r'<div class(.*)>(.*)</div>', '', content)
    content = re.sub(r'<br(.*?)/>', '', content)
    content = re.sub(r'<(.*)>(.*?)</(.*?)>', '', content)
    content = re.sub(r' \.', '.', content)

    # Adding Pokemon's bio to the list and notifying user of success
    bio.append(content)
    print("Completed text retrieval for {}".format(pokemon_names[i]))

# Storing the biological information on a CSV file
bio_data = pd.DataFrame(bio)
bio_data.to_csv('D:/UIP/scraping/pokemonbio.csv')

# Storing image urls on a CSV file for image retrieval in part 2
url_data = pd.DataFrame(imageurls)
url_data.to_csv('D:/UIP/scraping/pokemonimgurls.csv')


# Part 2 - image extraction

# Get list of Pokemon Names
df = pd.read_csv('D:/UIP/scraping/pokemonstats.csv', header=0)
pokemon_names = df['Name']

# Get Pokemon URLs with CDN directory
dfI = pd.read_csv('D:/UIP/scraping/pokemonimgurls.csv')
pokemon_images = dfI['0']

for i in range(802):

    # Define URL depending on Pokemon name and CDN folder structure
    URL = 'https://cdn.bulbagarden.net/upload/{}.png'.format(pokemon_images[i])

    # Stream image content from URL
    resp = requests.get(URL, stream=True)

    # Create a local file to store image contents
    pname = '{}.jpg'.format(pokemon_names[i])
    local_image = open(pname, 'wb')

    # Decoding image content
    resp.raw.decode_content = True

    # Storing the stream data on local image file
    shutil.copyfileobj(resp.raw, local_image)

    # Remove the image url response object.
    del resp

    # Prints success message
    print('Image retrieved for {}'.format(pname))


# Part 3 - Getting data for moves learnt by Pokemon

# Get list of Pokemon Names
df = pd.read_csv('D:/UIP/scraping/pokemonstats.csv', header=0)
pokemon_names = df['Name']

# List of moves to query for
# move_list = ['Bounce', 'Flamethrower', 'Ice_Beam', 'Thunderbolt', 'Sludge_Bomb', 'Iron_Head', 'Brick_Break', 'Dragon_Pulse', 'Absorb',
#              'Wing_Attack', 'Bite', 'Dazzling_Gleam', 'Confusion', 'Rock_Blast', 'Hypnosis', 'High_Jump_Kick', "Dark_Pulse", 'Mud_Shot', 'Scald', 'Bug_Bite']

move_list = ['Frost_Breath', 'Flame_Charge', 'Bug_Bite', 'Discharge', 'Metal_Claw', 'Psyshock', 'Draco_Meteor', 'Stealth_Rock', 'Magnitude', 'Foul_Play', 'Rock_Throw', 'Hex', 'Shadow_Sneak', 'Scald', 'Synthesis', 'Dazzling_Gleam', 'Wing_Attack', 'Close_Combat', 'High_Jump_Kick', 'Aurora_Veil', 'Shift_Gear']

# Array to store boolean values
move_data = np.zeros((len(pokemon_names), len(move_list)))

for j in range(len(move_list)):

    # Get Bulbapedia URL of that move
    URL = 'https://bulbapedia.bulbagarden.net/wiki/{}_(move)'.format(move_list[j])
    r = requests.get(URL)

    # Get a list of all Pokemon that learn that move
    imgloc = re.findall(
        r'<td style="text-align:center;" width="26px"> <a href="/wiki/(.*?)_', r.text)

    # Encode the corresponding column in the move_data array as 0 or 1
    for i in range(802):
        if pokemon_names[i] in imgloc:
            move_data[i, j] = 1

    # Prints success message
    print('Done for {}'.format(move_list[j]))

# Converts array to dataframe and stores as csv for future use
df = pd.DataFrame(move_data, columns=move_list)
df.to_csv('D:/UIP/scraping/pokemonmoves.csv')


# Part 4 - Creating the complete dataset

# Get list of Pokemon Names
df = pd.read_csv('D:/UIP/scraping/pokemonstats.csv', header=0)
pokemon_names = df['Name']
pokemon_type = df['Type1']
pokemon_typeB = df['Type2']

# Get data on biology and moves learnt
dfB = pd.read_csv('D:/UIP/scraping/pokemonbio.csv', index_col=0)
dfM = pd.read_csv('D:/UIP/scraping/pokemonmoves.csv', index_col=0)

# Combine all data for processing
data = pd.concat([pokemon_names, pokemon_type, pokemon_typeB, dfM, dfB], axis=1)
data = data.dropna(subset=['bio'])
data = data.set_index('Name')
data.to_csv('D:/UIP/scraping/pokemonfinal.csv')
