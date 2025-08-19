#!/bin/bash


LOG_FILE="pipeline.log"


> "$LOG_FILE"

# Log (change from overwriting to appending)
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}


run_command() {
    local command="$1"
    log "Running: $command"
    eval "$command" >> "$LOG_FILE" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        log "Error: Command failed with exit code $status - $command"
        return $status
    fi
    log "Success: $command"
    return 0
}


log "Starting pipeline..."


run_command "python bin.py" || { log "Pipeline failed at bin.py"; exit 1; }
run_command "python chr.py" || { log "Pipeline failed at chr.py"; exit 1; }
run_command "python sort.py" || { log "Pipeline failed at sort.py"; exit 1; }
run_command "python gk.py" || { log "Pipeline failed at gk.py"; exit 1; }
run_command "python all_gk.py" || { log "Pipeline failed at all_gk.py"; exit 1; }
run_command "python bg.py" || { log "Pipeline failed at bg.py"; exit 1; }


log "Pipeline completed successfully!"
exit 0