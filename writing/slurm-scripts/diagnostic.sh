#!/bin/bash
#
#SBATCH --job-name=diagnostic
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:05:00
#SBATCH --output=/tmp/diagnostic_%j.log
#SBATCH --error=/tmp/diagnostic_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

echo "========================================="
echo "DIAGNOSTIC REPORT"
echo "========================================="
echo ""

echo "1. Basic System Info:"
echo "   Hostname: $(hostname)"
echo "   Date: $(date)"
echo "   User: $(whoami)"
echo "   Home: $HOME"
echo "   PWD: $(pwd)"
echo ""

echo "2. Available Filesystems:"
df -h | head -20
echo ""

echo "3. AFS Access Test:"
if [ -d "/afs/csail.mit.edu/u/j/jhuang9" ]; then
    echo "   ✓ AFS directory is accessible"
    ls -ld /afs/csail.mit.edu/u/j/jhuang9/search-and-learn 2>&1
    echo "   Can list contents:"
    ls /afs/csail.mit.edu/u/j/jhuang9/search-and-learn 2>&1 | head -5
else
    echo "   ✗ AFS directory NOT accessible"
fi
echo ""

echo "4. Conda/Python Check:"
echo "   PATH: $PATH"
echo ""
echo "   Checking for conda:"
which conda 2>&1 || echo "   conda not in PATH"
echo ""
echo "   Checking for python:"
which python 2>&1 || echo "   python not in PATH"
which python3 2>&1 || echo "   python3 not in PATH"
echo ""
echo "   Checking specific conda path:"
ls -l /afs/csail.mit.edu/u/j/jhuang9/miniconda3/bin/conda 2>&1
echo ""
echo "   Checking conda.sh:"
ls -l /afs/csail.mit.edu/u/j/jhuang9/miniconda3/etc/profile.d/conda.sh 2>&1
echo ""

echo "5. Environment Variables:"
echo "   CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "   VIRTUAL_ENV: $VIRTUAL_ENV"
echo ""

echo "6. GPU Info:"
nvidia-smi 2>&1 || echo "   nvidia-smi not available"
echo ""

echo "7. Available Modules:"
module avail 2>&1 | head -20 || echo "   module system not available"
echo ""

echo "8. Try loading conda:"
if [ -f "/afs/csail.mit.edu/u/j/jhuang9/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "   Attempting to source conda.sh..."
    source /afs/csail.mit.edu/u/j/jhuang9/miniconda3/etc/profile.d/conda.sh 2>&1
    echo "   Result: $?"
    echo "   Conda available after sourcing:"
    which conda 2>&1
    echo ""
    echo "   Attempting to activate sal environment..."
    conda activate sal 2>&1
    echo "   Result: $?"
    echo "   Python after activation:"
    which python 2>&1
    python --version 2>&1
else
    echo "   ✗ conda.sh not found"
fi
echo ""

echo "9. Write Test:"
echo "   Testing write to /tmp:"
touch /tmp/test_write_$$ && echo "   ✓ Can write to /tmp" || echo "   ✗ Cannot write to /tmp"
rm -f /tmp/test_write_$$
echo ""
if [ -d "/afs/csail.mit.edu/u/j/jhuang9/search-and-learn" ]; then
    echo "   Testing write to AFS:"
    touch /afs/csail.mit.edu/u/j/jhuang9/search-and-learn/.test_write_$$ 2>&1 && echo "   ✓ Can write to AFS" || echo "   ✗ Cannot write to AFS"
    rm -f /afs/csail.mit.edu/u/j/jhuang9/search-and-learn/.test_write_$$ 2>&1
fi
echo ""

echo "========================================="
echo "END OF DIAGNOSTIC REPORT"
echo "========================================="
echo ""

# Try to copy log to a location we can access
echo "10. Attempting to copy log to accessible location:"
LOGFILE="/tmp/diagnostic_${SLURM_JOB_ID}.log"
if [ -d "/afs/csail.mit.edu/u/j/jhuang9/search-and-learn/writing/logs" ]; then
    cp "$LOGFILE" /afs/csail.mit.edu/u/j/jhuang9/search-and-learn/writing/logs/ 2>&1 && echo "   ✓ Copied to AFS" || echo "   ✗ Could not copy to AFS"
fi

# Also try copying to home
if [ -d "$HOME" ]; then
    cp "$LOGFILE" "$HOME/diagnostic_${SLURM_JOB_ID}.log" 2>&1 && echo "   ✓ Copied to HOME" || echo "   ✗ Could not copy to HOME"
fi
echo ""
echo "Log locations to check:"
echo "  - /tmp/diagnostic_${SLURM_JOB_ID}.log (on $(hostname))"
echo "  - ~/diagnostic_${SLURM_JOB_ID}.log"
echo "  - /afs/csail.mit.edu/u/j/jhuang9/search-and-learn/writing/logs/diagnostic_${SLURM_JOB_ID}.log"
