#include <stdio.h>
#include <stdlib.h>
#include <utmp.h>
#include <string.h>

/**
 * get_logged_in_users
 * Returns information about all currently logged-in users as a fixed-size string.
 *
 * The returned string is dynamically allocated and must be freed after use.
 */
char *get_logged_in_users() {
    struct utmp *entry;
    static const size_t MAX_BUFFER_SIZE = 1000; // Fixed buffer size
    char *result = (char *)malloc(MAX_BUFFER_SIZE);
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result[0] = '\0'; // Initialize the string
    size_t used_size = 0; // Tracks the current buffer usage

    // Initialize reading from the utmp file
    setutent();

    // Read entries from the utmp file
    while ((entry = getutent()) != NULL) {
        if (entry->ut_type == USER_PROCESS) { // Only process logged-in users
            char line[256];
            snprintf(line, sizeof(line), "%-16s %-16s %-16s\n",
                     entry->ut_user,
                     entry->ut_line,
                     entry->ut_host[0] ? entry->ut_host : "Local");

            size_t line_length = strlen(line);
            if (used_size + line_length >= MAX_BUFFER_SIZE) {
                // Stop if the buffer limit is exceeded
                fprintf(stderr, "Buffer limit exceeded: Cannot add more data.\n");
                break;
            }

            // Append the new line to the result string
            strcat(result, line);
            used_size += line_length;
        }
    }

    // Close the utmp file
    endutent();

    return result;
}


int main() {
    char *users = get_logged_in_users();
    if (users) {
        printf("Currently logged-in users:\n%s", users);
        free(users); // Free the allocated memory
    } else {
        printf("Failed to retrieve logged-in users.\n");
    }
    return 0;
}


