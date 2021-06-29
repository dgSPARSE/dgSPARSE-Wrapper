#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <errno.h>
#include <time.h>

#define log_enable 1
#define log_with_time 1
#define log_with_only_message 0

#define LOG_LEVEL INFO

#define TRACE 0
#define DEBUG 1
#define MEMORY_FOOTPRINT 7
#define INFO 8
#define WARNING 9
#define ERROR 10

#if (debug)
#define print_debug(format, args...) printf("%s:%d(%s) : " format, __FILE__, __LINE__, __func__, ##args)
#else
#define print_debug(...)
#endif

#if (log_enable)
#define LOG(level, format, args...)                                                                               \
    do                                                                                                            \
    {                                                                                                             \
        if (level >= LOG_LEVEL)                                                                                   \
        {                                                                                                         \
            char buffer[20];                                                                                      \
            buffer[0] = '\0';                                                                                     \
            if (log_with_time)                                                                                    \
            {                                                                                                     \
                time_t now = time(0);                                                                             \
                struct tm *timenow;                                                                               \
                timenow = localtime(&now);                                                                        \
                strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timenow);                                   \
            }                                                                                                     \
            if (log_with_only_message)                                                                            \
                printf("%s [%s] " format "\n", buffer, #level, ##args);                                           \
            else                                                                                                  \
                printf("%s [%s] %s:%d(%s) : " format "\n", buffer, #level, __FILE__, __LINE__, __func__, ##args); \
        }                                                                                                         \
    } while (0)
#else
#define log(...)
#endif

#endif // _LOGGER_H_