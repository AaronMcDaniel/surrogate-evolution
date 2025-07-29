#!/bin/bash
#SBATCH --job-name=auto_restart
#SBATCH --time=18:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Configuration
ORIGINAL_SCRIPT="main_ssi.job"
SCRIPT_ARGS_1='-o /storage/ice-shared/vip-vvk/data/AOT/psomu3/full_not_seeded -c conf.toml -n 100 -e nas -r'
SCRIPT_ARGS_2='-o /storage/ice-shared/vip-vvk/data/AOT/psomu3/full_no_pretrain -c conf_nopre.toml -n 100 -e nas -s seeds.txt -r'
RESTART_TIME_SECONDS=$((17 * 3600 + 45 * 60))  # 17h 30m in seconds
# RESTART_TIME_SECONDS=$(120)  # 2 minutes in seconds
MAX_RESTARTS=7
RESTART_COUNTER_FILE="restart_counter.txt"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Auto-restart orchestrator started (Job ID: $SLURM_JOB_ID) ==="

# Read current restart counter
if [[ -f "$RESTART_COUNTER_FILE" ]]; then
    restart_count=$(cat "$RESTART_COUNTER_FILE")
else
    restart_count=0
fi

log "Current restart iteration: $restart_count/$MAX_RESTARTS"

# Check if we've reached max restarts
if [[ $restart_count -ge $MAX_RESTARTS ]]; then
    log "Reached maximum restart limit ($MAX_RESTARTS). Stopping orchestrator."
    rm -f "$RESTART_COUNTER_FILE"
    exit 0
fi

# Submit both original scripts as separate SLURM jobs
log "Submitting first script: $ORIGINAL_SCRIPT $SCRIPT_ARGS_1"
original_job_id_1=$(sbatch --parsable $ORIGINAL_SCRIPT $SCRIPT_ARGS_1)

if [[ $? -ne 0 ]]; then
    log "ERROR: Failed to submit first original script!"
    exit 1
fi

log "First script submitted with Job ID: $original_job_id_1"

log "Submitting second script: $ORIGINAL_SCRIPT $SCRIPT_ARGS_2"
original_job_id_2=$(sbatch --parsable $ORIGINAL_SCRIPT $SCRIPT_ARGS_2)

if [[ $? -ne 0 ]]; then
    log "ERROR: Failed to submit second original script!"
    log "Cancelling first script: $original_job_id_1"
    scancel "$original_job_id_1"
    exit 1
fi

log "Second script submitted with Job ID: $original_job_id_2"

# Start timer
start_time=$(date +%s)

# Monitor until restart time
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    # Check if any other jobs (besides this orchestrator) are still running
    other_jobs=$(squeue -h -u "$USER" -o "%i" | grep -v "^$SLURM_JOB_ID$" | wc -l)
    if [[ $other_jobs -eq 0 ]]; then
        log "No other jobs running. Both original jobs completed."
        log "Orchestrator job completed successfully"
        rm -f "$RESTART_COUNTER_FILE"  # Clean up counter file
        exit 0
    fi
    
    # Check if it's time to restart
    if [[ $elapsed -ge $RESTART_TIME_SECONDS ]]; then
        log "Reached restart time (17h 30m). Initiating restart..."
        
        # Cancel all other jobs first
        log "Cancelling all other jobs..."
        other_job_ids=$(squeue -h -u "$USER" -o "%i" | grep -v "^$SLURM_JOB_ID$")
        for job_id in $other_job_ids; do
            log "Cancelling job: $job_id"
            scancel "$job_id"
        done
        
        # Increment restart counter for next iteration
        next_restart_count=$((restart_count + 1))
        echo "$next_restart_count" > "$RESTART_COUNTER_FILE"
        log "Updated restart counter to: $next_restart_count"
        
        # Submit next orchestrator job
        log "Submitting next orchestrator job..."
        next_orchestrator_id=$(sbatch --parsable "$0")
        
        if [[ $? -eq 0 ]]; then
            log "Next orchestrator submitted with Job ID: $next_orchestrator_id"
            
            # Wait 30 seconds for next orchestrator to start and submit its own original job
            log "Waiting 30 seconds for next orchestrator to initialize..."
            sleep 30
            
            # Cancel this orchestrator job (ourselves)
            log "Cancelling current orchestrator job: $SLURM_JOB_ID"
            scancel "$SLURM_JOB_ID"
            
            log "Handoff complete. Next orchestrator should continue the work."
            exit 0
        else
            log "ERROR: Failed to submit next orchestrator job!"
            exit 1
        fi
    fi
    
    # Check every 60 seconds
    sleep 60
done
