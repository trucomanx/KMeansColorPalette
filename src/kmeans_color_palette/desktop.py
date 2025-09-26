import os
import kmeans_color_palette.about as about
import subprocess


def update_desktop_database(desktop_path):
    applications_dir = os.path.expanduser(desktop_path)
    try:
        subprocess.run(
            ["update-desktop-database", applications_dir],
            check=True
        )
        print("Shortcut database updated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating the database: {e}")
    except FileNotFoundError:
        print("The command 'update-desktop-database' was not found. Verify that the package 'desktop-file-utils' is installed.")

def create_desktop_file(desktop_path, overwrite=False, program_name=None):
    base_dir_path = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_dir_path, 'icons', 'logo.png')

    if program_name is None:
        __program_name = about.__program_name__
    else:
        __program_name = program_name

    script_path = os.path.join("~",".local","bin",f"{__program_name}")
    script_path = os.path.expanduser(script_path)

    desktop_entry = f"""[Desktop Entry]
Name={__program_name}
Comment={about.__description__}
Exec={script_path}
Terminal=false
Type=Application
Icon={icon_path}
StartupNotify=true
Categories=Education;ResearchTools;
Keywords=organizer;python;
Encoding=UTF-8
StartupWMClass={about.__package__}
"""
    path = os.path.expanduser(os.path.join(desktop_path,f"{__program_name}.desktop"))
    
    if not os.path.exists(path) or overwrite == True: 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(desktop_entry)
        os.chmod(path, 0o755)
        print(f"File {__program_name}.desktop created in {path}.")
        update_desktop_database(desktop_path)
    
def create_desktop_directory(   directory_name = "ResearchTools",
                                long_name = "Scientific research",
                                comment = "Tools for Writing and Research Support",
                                icon = "accessories-text-editor", 
                                overwrite = False):
    
    desktop_entry = f"""[Desktop Entry]
Version=1.0
Type=Directory
Name={long_name}
Comment={comment}
Icon={icon}
"""
    
    path = os.path.join("~",".local","share","desktop-directories",f"{directory_name}.directory")
    path = os.path.expanduser(path)
    
    if not os.path.exists(path) or overwrite == True:  # Evita sobrescrever
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(desktop_entry)
        os.chmod(path, 0o755)
        print(f"File {path} created.")
    
def create_desktop_menu(directory_name = "ResearchTools",
                        basename = "research-tools",
                        overwrite = False):
    
    desktop_entry = f"""<!-- ~/.config/menus/applications-merged/{basename}.menu -->
<Menu>
    <Name>Applications</Name>
    <Menu>
        <Name>{directory_name}</Name>
        <Directory>{directory_name}.directory</Directory>
        <Include>
            <Category>{directory_name}</Category>
        </Include>
    </Menu>
</Menu>
"""
    path = os.path.join("~",".config","menus","applications-merged",f"{basename}.menu")
    path = os.path.expanduser(path)
    
    if not os.path.exists(path) or overwrite == True:  # Evita sobrescrever
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(desktop_entry)
        print(f"File {path} created.")

if __name__ == '__main__':
    create_desktop_menu()
    create_desktop_directory()
    create_desktop_file()

