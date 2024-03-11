# Compressive sensing project

Repository for the Compressive Sensing course projecr at CY Tech.

# Clone this Git repository

If you haven't already done so, configure your Git account:

```
git config --global user.name YourName
git config --global user.email MailOfYourGitHubAccount
```

Generate an SSH key: 
  - `ssh-keygen -t ed25519 -C "SSH key for cy_tech_compressive_sensing repository (https://github.com/cmnemoi/cy_tech_compressive_sensing)"`
  - Press `Enter` until the key is generated
- Add the SSH key to your SSH agent: `eval "$(ssh-agent -s)"" && ssh-add ~/(ssh-agent -s)". && ssh-add ~/.ssh/id_ed25519`
- Display the generated SSH key: `cat ~/.ssh/id_ed25519.pub` and copy it 
- Add the SSH key to your GitHub account:
  - Tutorial: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
  - Direct link: https://github.com/settings/ssh/new

Then clone this Git repository: 

`git clone git@github.com:cmnemoi/cy_tech_compressive_sensing .git && cd cy_tech_compressive_sensing` 

(enter `yes` if you're asked to confirm that the SSH key has been added to the list of known keys)

# Install requirements

You need [Python 3.12](https://www.python.org/downloads/) to run this project.

First, clone the project :
Then install the requirements with the following commands:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
