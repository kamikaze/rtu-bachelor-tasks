terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.84.0"
    }
  }

  backend "s3" {
    bucket = "rtu-infra"
    key    = "global/terraform.tfstate"
    region = "eu-north-1"
    profile = "rtu"
    dynamodb_table = "rtu_infra_tf_lockid"
  }

  required_version = ">= 1.10.0"
}
