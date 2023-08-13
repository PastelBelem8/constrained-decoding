ACTION=$1
REMOTE_PROJECT_DIR=cbelem@ava-s4.ics.uci.edu:/home/cbelem/projects/constrained-decoding
if [[$ACTION == 'pull']]
then
    scp -r $REMOTE_PROJECT_DIR .
    git status
else
    scp -r src $REMOTE_PROJECT_DIR
    scp setup.py $REMOTE_PROJECT_DIR
fi

git status