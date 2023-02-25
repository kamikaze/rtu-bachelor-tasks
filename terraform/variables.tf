variable "s3_enabled" {
  type    = bool
  default = true
}

variable "ecr_enabled" {
  type    = bool
  default = true
}

variable "ec2_enabled" {
  type    = bool
  default = false
}

variable "rds_enabled" {
  type    = bool
  default = false
}

variable "db_username" {
  type        = string
  nullable    = false
  default     = "bachelor"
  description = "Username for a postgres database"
}

variable "db_password" {
  type        = string
  nullable    = false
  description = "Password for a postgres database"

  validation {
    condition     = length(var.db_password) >= 18
    error_message = "Database password must be at least 18 characters long."
  }
}

variable "eks_enabled" {
  type    = bool
  default = false
}

variable "eks_cluster_name" {
  type    = string
  default = "rtu-bachelor-eks"
}
