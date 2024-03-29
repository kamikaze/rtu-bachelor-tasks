variable "organization" {
  type     = string
  default  = "Bixority"
  nullable = false

  validation {
    condition     = length(var.organization) >= 3
    error_message = "Organization name must be at least 3 characters long."
  }
}

variable "project" {
  type     = string
  default  = "rtu-bachelor-tasks"
  nullable = false

  validation {
    condition     = length(var.project) >= 3
    error_message = "Project name must be at least 3 characters long."
  }
}


variable "admin_ssh_public_key_name" {
  type     = string
  default  = "admin-ssh-key"
  nullable = false

  validation {
    condition     = length(var.admin_ssh_public_key_name) >= 3
    error_message = "Admin ssh key name must be at least 3 characters long."
  }
}

variable "key_pair_name" {
  type = string
}

variable "aws_region" {
  type     = string
  default  = "eu-north-1"
  nullable = false
}

variable "admin_ssh_public_key" {
  type     = string
  default  = ""
  nullable = false
}

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
  default = true
}

variable "db_name" {
  type        = string
  nullable    = false
  default     = "devdb"
  description = "Name for a postgres database"
}

variable "db_username" {
  type        = string
  nullable    = false
  default     = "dev"
  description = "Username for a postgres database"
}

variable "db_password" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "Password for a postgres database"

  validation {
    condition     = length(var.db_password) >= 18
    error_message = "Database password must be at least 18 characters long."
  }
}

variable "dataset_db_username" {
  type        = string
  nullable    = false
  default     = "dev"
  description = "Username for a dataset postgres database"
}

variable "dataset_db_password" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "Password for a dataset postgres database"

  validation {
    condition     = length(var.dataset_db_password) >= 18
    error_message = "Database password must be at least 18 characters long."
  }
}

variable "dns_api_token" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "DNS API token"
}

variable "dns_api_secret" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "DNS API key"
}

variable "eks_enabled" {
  type    = bool
  default = true
}

variable "eks_version" {
  type     = string
  default  = "1.29"
  nullable = false
}

variable "eks_cluster_name" {
  type     = string
  default  = "eks-dev"
  nullable = false
}

variable "eks_node_instance_type" {
  type     = string
  default  = "t4g.small"
  nullable = false
}

variable "github_token" {
  sensitive = true
  type      = string
}