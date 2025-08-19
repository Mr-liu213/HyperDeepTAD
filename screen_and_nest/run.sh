#!/bin/bash


LOG_FILE="pipeline.log"


> "$LOG_FILE"


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


run_command "python clustering_coefficient.py" || { log "Pipeline failed at clustering_coefficient.py"; exit 1; }
run_command "python screen.py" || { log "Pipeline failed at screen.py"; exit 1; }
run_command "python merge_TAD.py" || { log "Pipeline failed at merge_TAD.py"; exit 1; }

log "Pipeline completed successfully!"
exit 0
